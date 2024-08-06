To reproduce my work you'll need to do the following:

Get ahold of the Gemma 2 2B weights, specifically the layer 13 input layernorm, attention weights, and attention output layernorm parameters. These should be put in the "Weights and SAEs" folder and given their "default" names from the .safetensors file you'll get of Gemma 2 2B from huggingface.

Then you'll need the SAE weights, which you can get from Gemma Scope. These should be called "sae_res_layer_12.npz" and "sae_res_layer_13.npz"

Then just go into the get_qkov_circuits.py, decomment the functions in main(), and run it.
