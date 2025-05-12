"""
Ok so GLIDE has attention mechanisms throughout it and 

and we are going to analyze those attention components as the building blocks of the component graph


Here are the components of UNET
* input_blocks(Encoder blocks
* mid_blocks(Bottleneck)
* output_blocks(Decoder blocks

and eahc block contains:
- resBlock (for reidual connections
- attentionblock (text conditioning)
- downsampling/upsampling layers


1. first extracting components, in essence we say: Give me all attention heads as individual componeents

for block in self.model.input_blocks:
    for layer in block:
        if isinsance(layer,attentionblock)
            components.append(layer.qkv) # query, key. value matrices
            components.append(layer.proj_out) # output projection

2. then we ablate (set to zero) whatever components (randomized subset of the component graph)

def ablate(self, component index):
    attention = component_graph[index]
    orig_weights = attention.qkv.weight.clone()
    attention.qkv.weight.data = torch.zeros_like(orig_weigts)

3. for each ablation, check how much it impacts by computing the denoising error 
(this was suggested in SIM2024: " For diffusion-based generative models, one might study the
denoising error for a fixed timestep")

def compute_denoising_error(x_start, timestep,text_tokens,ablated_coponents):
    # add noise
    noise = torch.randn_like(x_start)
    s_noisy - self.diffusion.q_sample(x_start, timestep, noise)

    # ablate
    ablate(ablated_compoments)

    # denoise with ablated model
    predicted_noise = self.model(x_noisy, timestep. tokens= text_tokens)

    # measure error (mse)
    error = F.mse_loss(predicted_noise, noise)

    #restore
    restore_fucntion_iguess()

    return error

4. Build Linear model yayayay
ablation_vector = # binary vectors for what heads we keep vs ablate X n-thousand samples
error = #denoising error for each of these
reg -- Linearregression()
weights, bias = reg.fit(ablation_vectors, errors)

weights[i] = how much attention head i contribtues denoising

5. 

1. Get attributions for "a painting of a cat"
weights_painting = self.get_attributions(prompt1)

2. Get attributions for "a photograph of a cat"  
weights_photo = self.get_attributions(prompt2)

3. Find heads that differ
diff = weights_photo - weights_painting

4. Identify heads to modify
Positive diff = this head is more important for photographs
Negative diff = this head is more important for paintings
heads_to_ablate = np.where(diff < -0.05)[0]  # Strong painting heads

generate with those heads ablate for more=photographic output to model
"""