#  Keras implementation of Structural Deep Network Embedding, KDD 2016

- [The paper](http://www.kdd.org/kdd2016/subtopic/view/structural-deep-network-embedding)
- More details in [my blog article](http://xiaohan2012.github.io/2017/deep-structral-network-embedding-keras/)

# some examples

visualzing 20newsgroup data
- you can also [play with the embedding interactively](http://projector.tensorflow.org/?config=https://gist.githubusercontent.com/xiaohan2012/f9b66b262ba6f92b0f943be896338146/raw/b3cf61184380a435e710d1702a5f84b6fe6896b6/20news-projector-config.json)
  
![](http://xiaohan2012.github.io/assets/img/sdne/20newsgroup_viz.png)


label embedding from [stackexchange.datascience](https://datascience.stackexchange.com/) (`deep-learning` as an example)
- you can also [play with the embedding interactively](http://projector.tensorflow.org/?config=https://gist.githubusercontent.com/xiaohan2012/5c533ae2d4c67918c3648a23363307c6/raw/a23dd0b1540b3675d211e5f6db4ffdb969de202d/datascience-tensorboard-config)

![](http://xiaohan2012.github.io/assets/img/sdne/deep-learning.png)

# important scripts

## main algorithm

- `core.py`

## experiments

**20newsgroup visualization**

- `20newsgroup_train.py`: train for 20newsgroup dataset
- `20newsgroup_viz.py`: visualization using `sklearn.manifold.TSNE`
- `20newsgroup_tensorboard_embedding.py`: produce the embedding files for [tensorboard projector](https://www.tensorflow.org/versions/r0.12/how_tos/embedding_viz/), which is more interactive
  - you can also play with it [here](http://projector.tensorflow.org/?config=https://gist.githubusercontent.com/xiaohan2012/f9b66b262ba6f92b0f943be896338146/raw/b3cf61184380a435e710d1702a5f84b6fe6896b6/20news-projector-config.json) using trained embeddings


**link prediction**

- `link_prediction.py`: train (including grid search) and  test

**stackexchange label visualization**

- `stackexchange_train.py`: train for the stackexchange label cooccurence graph
- `stackexchange_label_embedding.py`: produce the embedding files for tensorboard projector
  - you can also play with it [here](http://projector.tensorflow.org/?config=https://gist.githubusercontent.com/xiaohan2012/5c533ae2d4c67918c3648a23363307c6/raw/a23dd0b1540b3675d211e5f6db4ffdb969de202d/datascience-tensorboard-config) using trained embeddings

# other implementations

- [suanrong/SDNE](https://github.com/suanrong/SDNE)
- [palash1992/GEM](https://github.com/palash1992/GEM)
