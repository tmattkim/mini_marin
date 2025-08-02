import jax.numpy as jnp
import jax
import numpy as np

def cross_entropy_loss(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
    return jnp.mean(loss)

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = jnp.sort(logits, axis=-1)[:, -k]
    return jnp.where(logits < values[:, None], -jnp.inf, logits)

def create_learning_rate_fn(config):
    import optax
    warmup_steps = 100
    peak_lr = 1e-3
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=config.num_epochs * 1000,
        end_value=1e-5,
    )
    return schedule

def beam_search_generate(model, params, input_ids, max_length, num_beams,
                         eos_token_id=None, temperature=1.0, top_k=None):
    batch_size = input_ids.shape[0]
    beam_size = num_beams
    input_ids = jnp.tile(input_ids[:, None, :], (1, beam_size, 1)).reshape(batch_size * beam_size, -1)

    beam_scores = jnp.zeros((batch_size, beam_size))
    sequences = input_ids

    for _ in range(max_length):
        logits = model.apply({"params": params}, sequences)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = jax.nn.softmax(logits, axis=-1)
        next_token_scores = jnp.log(probs)

        next_token_scores = beam_scores[:, :, None] + next_token_scores.reshape(batch_size, beam_size, -1)
        next_token_scores = next_token_scores.reshape(batch_size, -1)

        next_tokens = jnp.argsort(-next_token_scores, axis=-1)[:, :beam_size]
        beam_indices = next_tokens // logits.shape[-1]
        token_indices = next_tokens % logits.shape[-1]

        new_sequences = []
        for i in range(batch_size):
            seqs = []
            for beam in range(beam_size):
                seq = sequences[i * beam_size + beam_indices[i, beam]]
                seqs.append(jnp.concatenate([seq, jnp.array([token_indices[i, beam]])]))
            new_sequences.extend(seqs)

        sequences = jnp.stack(new_sequences)
        beam_scores = jnp.take_along_axis(next_token_scores, next_tokens, axis=1)

        if eos_token_id is not None:
            done = jnp.any(sequences[:, -1] == eos_token_id)
            if done:
                break

    return sequences.reshape(batch_size, beam_size, -1)[:, 0]  # best beam

def sample_logits(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = jax.nn.softmax(logits, axis=-1)
    return jax.random.categorical(jax.random.PRNGKey(np.random.randint(0, 10000)), logits)
