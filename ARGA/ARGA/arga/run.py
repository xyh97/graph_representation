import settings

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner
from node_prediction import Node_pred_Runner
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

dataname = 'pubmed'       # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
task = 'node_prediction'         # 'clustering' or 'link_prediction' or 'node_prediction'

settings = settings.get_settings(dataname, model, task)

if task == 'clustering':
    runner = Clustering_Runner(settings)
if task == 'node_prediction':
    runner = Node_pred_Runner(settings)
else:
    runner = Link_pred_Runner(settings)

runner.erun()

