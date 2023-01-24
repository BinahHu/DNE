import copy

import torch
from timm.models.layers import trunc_normal_
from torch import nn

import continual.utils as cutils
import continual.split_blocks_v2 as split_blocks_v2


class split_ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n

    def expand(self, extra_dim, extra_out):
        head = nn.Linear(self.embed_dim + extra_dim, self.nb_classes + extra_out, bias=True)
        if extra_out > 0:
            head.weight.data[:-extra_out, :-extra_dim] = self.head.weight.data
        else:
            head.weight.data[:, :-extra_dim] = self.head.weight.data

        norm = nn.LayerNorm(self.embed_dim + extra_dim)
        norm.weight.data[:-extra_dim] = self.norm.weight.data
        norm.bias.data[:-extra_dim] = self.norm.bias.data

        head.to(self.head.weight.device)
        norm.to(self.head.weight.device)
        self.head = head
        self.norm = norm
        self.nb_classes += extra_out
        self.embed_dim += extra_dim


class DyTox_Split(nn.Module):
    """"DyTox for the win!

    :param transformer: The base transformer.
    :param nb_classes: Thhe initial number of classes.
    :param individual_classifier: Classifier config, DyTox is in `1-1`.
    :param head_div: Whether to use the divergence head for improved diversity.
    :param head_div_mode: Use the divergence head in TRaining, FineTuning, or both.
    :param joint_tokens: Use a single TAB forward with masked attention (faster but a bit worse).
    """
    def __init__(
        self,
        transformer,
        nb_classes,
        individual_classifier='',
        head_div=False,
        head_div_mode=['tr', 'ft'],
        joint_tokens=False,
        single_token=False,
        split_block_config={}
    ):
        super().__init__()

        self.nb_classes = nb_classes
        self.embed_dim_list = transformer.embed_dim_list
        self.individual_classifier = individual_classifier
        self.use_head_div = head_div
        self.head_div_mode = head_div_mode
        self.head_div = None
        self.joint_tokens = joint_tokens
        self.in_finetuning = False

        self.nb_classes_per_task = [nb_classes]

        self.patch_embed = transformer.patch_embed
        self.num_patches = transformer.num_patches
        self.pos_embed_list = transformer.pos_embed_list
        self.pos_drop = transformer.pos_drop
        self.sabs = transformer.blocks[:transformer.local_up_to_layer]

        self.tabs = transformer.blocks[transformer.local_up_to_layer:]
        self.split = split_block_config['split']

        self.single_token = single_token
        self.task_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, self.embed_dim).cuda())])

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                split_ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = split_ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()

    @property
    def embed_dim(self):
        return sum(self.embed_dim_list)

    @property
    def pos_embed(self):
        return torch.cat(list(self.pos_embed_list), dim=-1)

    def end_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def freeze_split_old(self):
        self.patch_embed.freeze_split_old()
        for p in self.pos_embed_list[:-1]:
            p.requires_grad = False
        for b in self.sabs:
            b.freeze_split_old()

        for b in self.tabs:
            b.freeze_split_old()

    def fix_and_update_attn(self):
        self.apply(self.module_fix_and_update_attn)

    def module_fix_and_update_attn(self, m):
        if isinstance(m, split_blocks_v2.split_Linear):
            m.fix_and_update_attn()

    def expand(self, extra_dim, extra_heads, increment):
        self.nb_classes_per_task.append(increment)
        self.embed_dim_list.append(extra_dim)

        self.patch_embed.expand(extra_dim)

        if not self.split:
            new_pos_embd = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim).cuda())
            new_pos_embd.data[..., :-extra_dim] = self.pos_embed_list[-1].data
            self.pos_embed_list[-1] = new_pos_embd
        else:
            for p in self.pos_embed_list.parameters():
                p.requires_grad = False
            self.pos_embed_list.append(nn.Parameter(torch.zeros(1, self.num_patches, extra_dim).cuda()))
            trunc_normal_(self.pos_embed_list[-1], std=.02)

        for b in self.sabs:
            b.expand(extra_dim, extra_heads)

        for b in self.tabs:
            b.expand(extra_dim, extra_heads)

        if self.single_token:
            new_task_token = nn.Parameter(torch.zeros(1, 1, extra_dim).cuda())
            trunc_normal_(new_task_token, std=.02)
            self.task_tokens.append(new_task_token)
        else:
            new_task_tokens = nn.ParameterList([])
            for i in range(len(self.nb_classes_per_task)):
                new_task_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim).cuda())
                ref_id = i if i < len(self.nb_classes_per_task) - 1 else -1
                new_task_token.data[:,:,:-self.embed_dim_list[-1]] = self.task_tokens[ref_id].data
                new_task_tokens.append(new_task_token)
            del self.task_tokens
            self.task_tokens = new_task_tokens

        if self.use_head_div:
            self.head_div = split_ContinualClassifier(
                self.embed_dim, self.nb_classes_per_task[-1] + 1
            ).cuda()

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            for head in self.head:
                head.expand(extra_dim, 0)
            self.head.append(
                split_ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            if self.single_token:
                in_dim = self.embed_dim
            else:
                in_dim = self.embed_dim * len(self.task_tokens)
            self.head = split_ContinualClassifier(
                in_dim, sum(self.nb_classes_per_task)
            ).cuda()

    def add_model(self, nb_new_classes):
        """Expand model as per the DyTox framework given `nb_new_classes`.

        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        # Class tokens ---------------------------------------------------------
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        # ----------------------------------------------------------------------

        # Diversity head -------------------------------------------------------
        if self.use_head_div:
            self.head_div = split_ContinualClassifier(
                self.embed_dim, self.nb_classes_per_task[-1] + 1
            ).cuda()
        # ----------------------------------------------------------------------

        # Classifier -----------------------------------------------------------
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head.append(
                split_ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            self.head = split_ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()
        # ----------------------------------------------------------------------

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.

        By default, DyTox is in 1-1.
        """
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            if self.single_token:
                in_dim = self.embed_dim
            else:
                in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            if self.single_token:
                in_dim = self.embed_dim
            else:
                in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()
        self.freeze_split_old()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.task_tokens, requires_grad=requires_grad)
            elif name == 'sab':
                self.sabs.eval()
                cutils.freeze_parameters(self.patch_embed, requires_grad=requires_grad)
                cutils.freeze_parameters(self.pos_embed_list, requires_grad=requires_grad)
                cutils.freeze_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.freeze_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': self.pos_embed_list.parameters(),
            'ca': self.tabs.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                              if self.individual_classifier else \
                              self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
            'head_div': self.head_div.parameters() if self.head_div is not None else None
        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.task_tokens)):
                for j in range(i + 1, len(self.task_tokens)):
                    if self.single_token:
                        dist = 0
                    else:
                        dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        return log

    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses

    @torch.no_grad()
    def check(self):
        x = torch.ones(2, 3, 32, 32).cuda()
        B = x.shape[0]

        x = self.patch_embed(x)
        print("Patch embd")
        print(x[0, 0, :10])
        x = x + self.pos_embed
        print("Pos embd")
        print(x[0, 0, :10])
        x = self.pos_drop(x)

        s_e, s_a, s_v = [], [], []
        for blk in self.sabs:
            x, attn, v = blk(x)
            s_e.append(x)
            s_a.append(attn)
            s_v.append(v)
        print("SAB")
        print(x[0, 0, :10])

        # Specific part, this is what we called the "task specific DECODER"
        if self.joint_tokens:
            return self.forward_features_jointtokens(x)

        tokens = []
        attentions = []
        mask_heads = None

        if self.single_token:
            task_token = torch.cat(list(self.task_tokens), dim=-1)
            task_token = task_token.expand(B, -1, -1)
            print("Task token")
            print(task_token[0, 0, :10])

            for blk in self.tabs:
                task_token, attn, v = blk(torch.cat((task_token, x), dim=1), mask_heads=mask_heads)
            attentions = [attn]
            tokens = [task_token[:, 0]]
            print("Display final token")
            print(tokens[0][0, :10])
            c = input()
        else:
            for task_token in self.task_tokens:
                task_token = task_token.expand(B, -1, -1)

                for blk in self.tabs:
                    task_token, attn, v = blk(torch.cat((task_token, x), dim=1), mask_heads=mask_heads)

                attentions.append(attn)
                tokens.append(task_token[:, 0])

        self._class_tokens = tokens


    def forward_features(self, x):
        # Shared part, this is the ENCODER
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        s_e, s_a, s_v = [], [], []
        for blk in self.sabs:
            x, attn, v = blk(x)
            s_e.append(x)
            s_a.append(attn)
            s_v.append(v)

        # Specific part, this is what we called the "task specific DECODER"
        if self.joint_tokens:
            return self.forward_features_jointtokens(x)

        tokens = []
        attentions = []
        mask_heads = None


        if self.single_token:
            task_token = torch.cat(list(self.task_tokens), dim=-1)
            task_token = task_token.expand(B, -1, -1)

            for blk in self.tabs:
                task_token, attn, v = blk(torch.cat((task_token, x), dim=1), mask_heads=mask_heads)
            attentions = [attn]
            tokens = [task_token[:, 0]]
        else:
            for task_token in self.task_tokens:
                task_token = task_token.expand(B, -1, -1)

                for blk in self.tabs:
                    task_token, attn, v = blk(torch.cat((task_token, x), dim=1), mask_heads=mask_heads)

                attentions.append(attn)
                tokens.append(task_token[:, 0])

        self._class_tokens = tokens
        return tokens, tokens[-1], attentions

    def forward_features_jointtokens(self, x):
        """Method to do a single TAB forward with all task tokens.

        A masking is used to avoid interaction between tasks. In theory it should
        give the same results as multiple TAB forward, but in practice it's a little
        bit worse, not sure why. So if you have an idea, please tell me!
        """
        raise NotImplementedError("This function not implemented yet")
        B = len(x)

        task_tokens = torch.cat(
            [task_token.expand(B, 1, -1) for task_token in self.task_tokens],
            dim=1
        )

        for blk in self.tabs:
            task_tokens, _, _ = blk(
                torch.cat((task_tokens, x), dim=1),
                task_index=len(self.task_tokens),
                attn_mask=True
            )

        if self.individual_classifier in ('1-1', '1-n'):
            return task_tokens.permute(1, 0, 2), task_tokens[:, -1], None
        return task_tokens.view(B, -1), task_tokens[:, -1], None

    def forward_classifier(self, tokens, last_token):
        """Once all task embeddings e_1, ..., e_t are extracted, classify.

        Classifier has different mode based on a pattern x-y:
        - x means the number of task embeddings in input
        - y means the number of task to predict

        So:
        - n-n: predicts all task given all embeddings
        But:
        - 1-1: predict 1 task given 1 embedding, which is the 'independent classifier' used in the paper.

        :param tokens: A list of all task tokens embeddings.
        :param last_token: The ultimate task token embedding from the latest task.
        """
        logits_div = None

        if self.individual_classifier != '':
            logits = []

            for i, head in enumerate(self.head):
                if self.single_token:
                    logits.append(head(last_token))
                elif self.individual_classifier in ('1-n', '1-1'):
                    logits.append(head(tokens[i]))
                else:  # n-1, n-n
                    logits.append(head(torch.cat(tokens[:i+1], dim=1)))

            if self.individual_classifier in ('1-1', 'n-1') or self.single_token:
                logits = torch.cat(logits, dim=1)
            else:  # 1-n, n-n
                final_logits = torch.zeros_like(logits[-1])
                for i in range(len(logits)):
                    final_logits[:, :logits[i].shape[1]] += logits[i]

                for i, c in enumerate(self.nb_classes_per_task):
                    final_logits[:, :c] /= len(self.nb_classes_per_task) - i

                logits = final_logits
        elif isinstance(tokens, torch.Tensor):
            logits = self.head(tokens)
        else:
            logits = self.head(torch.cat(tokens, dim=1))

        if self.head_div is not None and eval_training_finetuning(self.head_div_mode, self.in_finetuning):
            logits_div = self.head_div(last_token)  # only last token

        return {
            'logits': logits,
            'div': logits_div,
            'tokens': tokens
        }

    def forward(self, x):
        tokens, last_token, _ = self.forward_features(x)
        return self.forward_classifier(tokens, last_token)


def eval_training_finetuning(mode, in_ft):
    if 'tr' in mode and 'ft' in mode:
        return True
    if 'tr' in mode and not in_ft:
        return True
    if 'ft' in mode and in_ft:
        return True
    return False
