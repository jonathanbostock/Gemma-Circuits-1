### Jonathan Bostock

import numpy as np
import pandas as pd
import torch
import jbplot
import json

def main():

    # Old group of functions which plot the QK-circuit between the decoder bias
    # and itself, as a function of RoPE. This was to estimate the effect of position
    # on the attention heads. Unclear interpretation and I decided to give up and
    # ignore it
    """
    position_set = np.array([0] + list(np.round(np.exp(np.linspace(0, np.log(8192), 50)))))
    save_qk_values(position_set)
    plot_qk_values(position_set)
    """

    # save_qkov_values()
    ## This saves query, key, value, and pre-output vectors for each feature

    # save_qk_ov_matrices()
    ## This saves the (large!) qk_ov matrices

    # generate_tables()
    ## This generates and saves .csv files of the largest QK and OV connection values

    # deduplicate_qk_csvs()
    ## This creates QK .csv files with the QK-like-to-like connections removed

def deduplicate_qk_csvs():

    for head in range(8):

        df = pd.read_csv(f"Outputs/Table of QKs for Head {head}.csv",
                         index_col = 0)

        new_df = df.loc[df["Query Feature"] != df["Key Feature"]]
        new_df.index = range(len(new_df))
        new_df.to_csv(f"Outputs/Table of Distinct QKs for Head {head}.csv")

        new_df_pos = new_df[new_df["QK Circuit Value"] > 0]
        new_df_pos.index = range(len(new_df_pos))
        new_df_pos.to_csv(f"Outputs/Table of Distinct Positive QKs for Head {head}.csv")

def generate_tables(heads=range(8), max_items=1000, num_ov_examples=20):

    decoder_labels = get_features_list("Weights and SAEs/layer_12_res_sae.json")
    encoder_labels = get_features_list("Weights and SAEs/layer_13_res_sae.json")

    size = len(decoder_labels)-1


    # This shouldn't normally run
    if "sum" in heads:
        qk_sum_matrix = np.zeros(shape = (size, size)).astype(np.float16)
        ov_sum_matrix = np.zeros(shape = (size, size)).astype(np.float16)

    for head in list(heads):

        ## Ignore this lol
        if head == "sum":
            qk_matrix = qk_sum_matrix
            ov_matrix = ov_sum_matrix
        else:
            ## First let's get the strongest QK-circuits
            qk_matrix = np.load(f"Saved Numpy Matrices/head_{head}_qk_matrix.npy").astype(np.float16)
            ov_matrix = np.load(f"Saved Numpy Matrices/head_{head}_ov_matrix.npy").astype(np.float16)

            if "sum" in heads:
                qk_sum_matrix += qk_matrix
                ov_sum_matrix += ov_matrix


        abs_qk_locations = largest_n_indices(np.abs(qk_matrix), max_items)
        qk_locations = largest_n_indices(qk_matrix, max_items)

        # Use the absolute value here
        qk_values = [[qk_matrix[qk[0], qk[1]],
                      f"{qk[1]}: {decoder_labels[qk[1]]}",
                      f"{qk[0]}: {decoder_labels[qk[0]]}"]
                     for qk in abs_qk_locations]

        qk_dataframe = pd.DataFrame(qk_values, columns=["QK Circuit Value",
                                                        "Key Feature",
                                                        "Query Feature"])
        qk_dataframe.to_csv(f"Outputs/Table of QKs for Head {head}.csv")

        ## Now let's get the strongest OV-circuits
        ov_locations = largest_n_indices(np.abs(ov_matrix), max_items)

        vo_values = [[ov_matrix[ov[0], ov[1]],
                      f"{ov[1]}: {decoder_labels[ov[1]]}",
                      f"{ov[0]}: {encoder_labels[ov[0]]}"]
                     for ov in ov_locations]

        vo_dataframe = pd.DataFrame(vo_values, columns=["OV Circuit Value",
                                                        "Value Feature",
                                                        "Output Feature"])
        vo_dataframe.to_csv(f"Outputs/Table of OVs for Head {head}.csv")

        ## This next part is deprecated, it was just very confusing
        """
        ## Finally, let's make a full pivot table
        qkov_values = []
        for qk in qk_locations:
            kq = [decoder_labels[qk[1]], decoder_labels[qk[0]]]

            ov_row = ov_matrix[...,qk[1]]

            ## Get the argmaxes for our ov_row
            max_ov_indices = np.argpartition(
                np.abs(ov_row), -num_ov_examples)[-num_ov_examples:][::-1]

            for max_ov_i in max_ov_indices:
                row = [ov_row[max_ov_i], *kq, encoder_labels[max_ov_i]]
                qkov_values.append(row)

        qkov_dataframe = pd.DataFrame(qkov_values,
                                      columns = ["QKOV", "Key", "Query", "Value"])

        qkov_dataframe.sort_values(by=['QKOV'], key = abs, ascending=False, inplace=True)
        qkov_dataframe.index = range(len(qkov_dataframe))
        qkov_dataframe.iloc[0:max_items].to_csv(
            f"Outputs/Table of QKOVs for Head {head}.csv")
        """

        ## Plot correlation coefficient
        qk_by_k = np.mean(qk_matrix, axis=0)
        ov_by_v = np.mean(np.abs(ov_matrix), axis=0)

        qk_ov_correlation = np.corrcoef(x=qk_by_k, y=ov_by_v)[0,1]

        fig, ax = jbplot.figax()
        ax.scatter(qk_by_k, ov_by_v)

        ax.set_title(f"Head {head}\nCorrelation = {qk_ov_correlation:.2f}")
        ax.set_xlabel("Average QK Circuit Value by Sender Feature")
        ax.set_ylabel("Average Absolute OV Circuit Value by Sender Feature")

        jbplot.save(fig, f"Outputs/Head {head} QK-OV Correlation",
                    file_types = ["png"])



def largest_n_indices(arr, n):
    # Flatten the array and get the indices of the n largest elements
    flat_indices = np.argpartition(arr.ravel(), -n)[-n:]

    # Sort these indices based on their corresponding values (in descending order)
    flat_indices = flat_indices[np.argsort(-arr.ravel()[flat_indices])]

    # Convert flat indices to 2D indices and return as nx2 array
    return np.array(np.unravel_index(flat_indices, arr.shape)).T


def get_features_list(file_name):

    explanations = json.load(open(file_name, "r"))["explanations"]

    return_list = [""] * (1 << 14)

    for dict_ in explanations:

        return_list[int(dict_["index"])] = dict_["description"]

    return return_list


def save_qk_ov_matrices(matrix_size = -1, heads = range(8)):

    queries = np.load("Saved Numpy Matrices/queries.npy")[:matrix_size].astype(np.float16)
    keys = np.load("Saved Numpy Matrices/keys.npy")[:matrix_size].astype(np.float16)
    values = np.load("Saved Numpy Matrices/values.npy")[:matrix_size].astype(np.float16)
    outputs = np.load("Saved Numpy Matrices/outputs.npy")[:matrix_size].astype(np.float16)

    for head in heads:

        qo_start = 256 * head
        kv_start = 256 * (head // 2)

        head_qs = queries[::,qo_start:qo_start+256]
        head_ks = keys[::,kv_start:kv_start+256]

        head_os = outputs[::,qo_start:qo_start+256]
        head_vs = values[::,kv_start:kv_start+256]

        qk_matrix = np.einsum("qd,kd->qk", head_qs, head_ks)
        np.save(open(f"Saved Numpy Matrices/head_{head}_qk_matrix.npy", "wb"), qk_matrix)

        ov_matrix = np.einsum("od,vd->ov", head_os, head_vs)
        np.save(open(f"Saved Numpy Matrices/head_{head}_ov_matrix.npy", "wb"), ov_matrix)



def save_qkov_values():
    sae_12 = np.load("Weights and SAEs/sae_res_layer_12.npz")
    sae_13 = np.load("Weights and SAEs/sae_res_layer_13.npz")

    decoder_features = sae_12["W_dec"]
    encoder_features = sae_13["W_enc"].T

    norm_input = torch.load("Weights and SAEs/model.layers.13.input_layernorm.weight.pt").numpy()
    norm_output = torch.load("Weights and SAEs/model.layers.13.post_attention_layernorm.weight.pt").numpy()
    W_q = torch.load("Weights and SAEs/model.layers.13.self_attn.q_proj.weight.pt").numpy()
    W_k = torch.load("Weights and SAEs/model.layers.13.self_attn.k_proj.weight.pt").numpy()
    W_o = torch.load("Weights and SAEs/model.layers.13.self_attn.o_proj.weight.pt").numpy()
    W_v = torch.load("Weights and SAEs/model.layers.13.self_attn.v_proj.weight.pt").numpy()

    decoder_normed = decoder_features * (1 + norm_input)
    encoder_normed = encoder_features * (1 + norm_output)

    queries = decoder_normed @ W_q.T
    keys = decoder_normed @ W_k.T
    values = decoder_normed @ W_v.T
    outputs = encoder_normed @ W_o

    np.save(open("Saved Numpy Matrices/queries.npy", "wb"), queries)
    np.save(open("Saved Numpy Matrices/keys.npy", "wb"), keys)
    np.save(open("Saved Numpy Matrices/values.npy", "wb"), values)
    np.save(open("Saved Numpy Matrices/outputs.npy", "wb"), outputs)


def plot_qk_values(position_set):

    qk_values = np.load("Saved Numpy Matrices/qk_values_position.npy")

    qk_values_adjusted = (qk_values.T - np.mean(qk_values, axis=1)).T

    fig, ax = jbplot.figax()
    jbplot.plotlineset(ax, x_vect_set = [position_set],
                       y_vect_set=qk_values_adjusted,
                       gradient=True,
                       name_list =[f"Head {i}" for i in range(8)])

    jbplot.nice_legend(ax)
    ax.set_xlabel("Token Distance")
    ax.set_ylabel("Attention value of decoder bias")
    ax.set_xscale("symlog")

    jbplot.save(fig, "Outputs/Attention Value of Decoder Bias",
                file_types=["png"])


def save_qk_values(position_set):
    sae = np.load("sae_res_layer_12.npz")

    bias = sae["b_dec"]

    norm = torch.load("Weights and SAEs/model.layers.13.input_layernorm.weight.pt").numpy()
    W_q = torch.load("Weights and SAEs/model.layers.13.self_attn.q_proj.weight.pt").numpy()
    W_k = torch.load("Weights and SAEs/model.layers.13.self_attn.k_proj.weight.pt").numpy()

    bias_normed = bias * (norm + 1)

    bias_q = bias_normed @ W_q.T
    bias_k = bias_normed @ W_k.T


    qk_values = np.zeros((8, len(position_set)), np.float16)

    for head in range(8):

        k_head = head // 2

        query = bias_q[head*256:(head+1)*256]
        key = bias_k[k_head*256:(k_head+1)*256]

        for i, position in enumerate(position_set):

            qk_values[head, i] = np.dot(key, rope(x = query, k = position)) / np.sqrt(256)

    np.save(open("Saved Numpy Matrices/qk_values_position.npy", "wb"), qk_values)


def rope(x: np.ndarray, k: int, head_dim: int = 256, base: float = 10000.0) -> np.ndarray:
    """
    Applies rotary position embedding to the input array for position k.
    Args:
        x: Input array of shape (..., head_dim)
        k: Position to encode
        head_dim: Dimension of each head (default 256 based on the model)
        max_seq_len: Maximum sequence length (default 8192, typical for many models)
        base: Base for the angle calculation (default 10000.0, as used in the original implementation)

    Returns:
    Rotary position embedded array of the same shape as x
    """
    # Ensure head_dim is even
    assert head_dim % 2 == 0, "head_dim must be even"

    # Generate position encoding
    theta = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    position = np.array([k])  # We're only encoding for position k
    freqs = position[:, np.newaxis] * theta[np.newaxis, :]

    # Create complex numbers for rotation
    cos_theta = np.cos(freqs)
    sin_theta = np.sin(freqs)

    # Reshape x into complex numbers
    x_complex = x[..., ::2] + 1j * x[..., 1::2]

    # Apply rotation
    x_rotated = x_complex * (cos_theta + 1j * sin_theta)

    # Convert back to real numbers
    x_out = np.stack((x_rotated.real, x_rotated.imag), axis=-1).reshape(x.shape)

    return x_out

if __name__ == "__main__":

    main()
