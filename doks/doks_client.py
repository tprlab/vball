import requests
import os
import json
import time

do_token = os.environ.get("DO_TOKEN")
cluster_name = os.environ.get("DO_CLUSTER")
nodepool_name = os.environ.get("DO_NODEPOOL")

machine_type = os.environ.get("MACHINE_TYPE")
if machine_type is None:
  machine_type = "c-2"


do_url="https://api.digitalocean.com/v2/"
do_headers = {'Content-type': 'application/json', "Authorization" : "Bearer {0}".format(do_token)}

class DoError(Exception):
  pass 

def do_get(url):
  rsp = requests.get(url, headers=do_headers)
  if rsp.status_code != requests.codes.ok:
    raise DoError(rsp.status_code, rsp.content)

  return rsp.json()

def do_post(url, params = None):
  return requests.post(url, data=json.dumps(params), headers=do_headers)

def do_delete(url):
  return requests.delete(url, headers=do_headers)



def get_clusters():
  url = os.path.join(do_url, "kubernetes/clusters")
  return do_get(url)

def find_cluster_by_name(cl, name):
  for c in cl["kubernetes_clusters"]:
    if c["name"] == name:
      return c
  return None

def list_node_pools(cluster):
  url = os.path.join(do_url, "kubernetes/clusters", cluster, "node_pools")
  rsp = do_get(url)
  return rsp["node_pools"]


def create_node_pool(cluster_id, name, size, count=1, labels=None):
  data = {"name" : name, "size" : size, "count" : count, "labels" : labels}
  url = os.path.join(do_url, "kubernetes/clusters", cluster_id, "node_pools")
  rsp = do_post(url, data)
  if rsp.status_code != requests.codes.created:
    raise DoError(rsp.status_code, rsp.content)
  return rsp.json()


def get_first_node(nodepool):
  for n in nodepool["nodes"]:
    return n
  return None

def get_node_pool(cluster, nodepool):
  url = os.path.join(do_url, "kubernetes/clusters", cluster, "node_pools", nodepool)
  rsp = do_get(url)
  return rsp["node_pool"]

def find_node_pool(cluster, name):
  node_pools = list_node_pools(cluster)
  for n in node_pools:
    if n["name"] == name:
      return n
  return None


def have_node_pool_ready(cluster, name, size, count=1, labels=None):
  nodepool = find_node_pool(cluster, name)
  if nodepool is None:
    rsp = create_node_pool(cluster, name, size,count, labels)
    nodepool = rsp["node_pool"]
    nodepool_id = nodepool["id"]
    node = get_first_node(nodepool)
    print("Nodepool created", nodepool_id, "node", node["id"], node["status"])
    nodepool = find_node_pool(cluster, name)

  for i in range(30):
    node = get_first_node(nodepool)
    status = node["status"]["state"]
    if status == "running":
      print ("Node ready")
      return nodepool["id"]
    print(i, "Node", node["id"], status)
    time.sleep(10)
    nodepool = find_node_pool(cluster, name)
    if nodepool is None:
      print ("Node disappered")
      return None

  return None


def delete_node_pool(cluster, nodepool): 
  url = os.path.join(do_url, "kubernetes/clusters", cluster, "node_pools", nodepool)
  return do_delete(url)


if __name__ == "__main__":
  clusters = get_clusters()
  vb_cluster = find_cluster_by_name(clusters, cluster_name)
  print(vb_cluster)
  nodepool_id = have_node_pool_ready(vb_cluster["id"], nodepool_name, machine_type, 1, {"vball-target": "vproc"})

  if nodepool_id:
    time.sleep(5)
    rsp = delete_node_pool(vb_cluster["id"], nodepool_id)
    print ("Deleted node", nodepool_id, rsp)




