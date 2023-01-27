# attnviz

Overview:
- This repo contains code for visualizing the attention patterns of standard decoder-only GPT-style models in an interactive jupyter notebook. The demo notebook has an example for visualizing attention in GPT-J (https://huggingface.co/docs/transformers/model_doc/gptj)

- While other methods exist for visualizing attention patterns of larger language models, they don't scale well for longer prompts/contexts. I built two widgets (the `AttnHeadSelectorWidget`, and `TokenVizWidget`) that interact to help solve this problem.

<p>
  <img src="https://github.com/ericwtodd/attnviz/blob/1ef059b0ed3526328e484f44afee2348d91a4a51/images/restaurant_demo_small.gif" height="350"/>
</p>

Usage:
- Use the selectors to pick an attention head. Hover over a token to display its attention pattern and tooltip.
- Click on a token to lock its attention pattern. Click the same token to unlock. 
- If no token is hover-selected or clicked, the default display is the contents of the `default_matrix` array passed to the `TokenVizWidget`.


Geting Started:
- This tool runs inside jupyter notebook. I've made a simple conda environment that supports this visualization tool which you can install to then open jupyter notebook and get started using the following commands:

```
conda env create -f attnviz.yml
conda activate attnviz
jupyter notebook
```
