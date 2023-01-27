# attnviz

Overview:
- This repo contains code for visualizing the attention patterns of standard decoder-only GPT-style models in an interactive jupyter notebook. (The demo notebook has an example for visualizing attention in GPT-J (https://huggingface.co/docs/transformers/model_doc/gptj)

- While other methods exist for visualizing attention patterns of larger language models, they don't scale well for longer prompts/contexts.

- There are two widgets I built that interact to solve this problem. The `AttnHeadSelectorWidget`, and `TokenVizWidget`.

Usage:
- Use the selectors to pick an attention head. Hover over a token to display its attention pattern and tooltip.
- Click on a token to lock its attention pattern. Click the same token to unlock. 
- If no token is hover-selected or clicked, the default display is the contents of the `default_matrix` array passed to the `TokenVizWidget`.

<!-- Insert GIF of the tool here -->


Get Started:
- You can install the conda environment that supports this visualization tool with the following command:

```
conda env create -f attnviz.yml
```





Logistics:
- GPT-J has 28 layers with 16 heads per layer.<br>
- There are two widgets the attn head selector widget and the token viz widget. <br>
- The `TokenVizWidget` requires the tokenized prompt text, `attention matrix` of size `(n_layers,n_heads,n_tokens,n_tokens)`, and a `default matrix` of size `(n_layers,n_heads,n_tokens)` to build.<br> 
- Unless you have a specific default you're interested in, you can just create an `ndarray` of 0's to fill the slot.
