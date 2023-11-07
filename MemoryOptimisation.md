# Memory Optimisation Techniques in PyTorch

Memory management can often be the main point of frustration when writing deep learning 
models. These problems are often magnified when training with large data samples or creating
a model with large and complex architecture. Here are some of the potential reasons for 'out-of-memory'
errors and some ways of avoiding them.

### Data Size
A very simple way of minimising memory usage is to reduce the amount of data flowing through the
network at any given time. The simplest way of doing this is to reduce the batch size. Another,
but not as universally applicable, method is to scale down the size of each training sample (this
really only applies to experimental models).

If large batch sizes are necessary, gradient aggregation (or accumulation) can be used to minimise
memory overhead. Instead of recomputing weights after every backward pass, gradients can be aggregated
across `n` number of batches and then weights can be calculated in a single update.
``` python:
if batch_num+1 % n == 0:
    optimiser.step()
    optimiser.zero_grad()
```

### Model Architecture
Complex or 'deeper' models with many layers will consume more memory trying store the model's parameters.
Reducing the memory used by these complex models is less straightforward.

Model pruning removes nodes from a network, based on specified criteria. Here two examples of 
pruning the connections in a given layer:

```python:
# remove 20% of the connections in the parameter "weight" for a given layer
prune.random_unstructured(module, name="weight", amount=0.2)

# remove the 5 smallest entries using L1 norm in the parameter "bias" for a given layer
prune.l1_unstructured(module, name="bias", amount=5)
```

The parameters from a larger pre-trained model can be used to initialise another model that is
 going to be trained for a more specialised purpose. This technique is often known as "knowledge
distillation".

