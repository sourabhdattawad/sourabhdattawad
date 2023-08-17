---
layout: post
title:  "Machine Learning System Testing: A Guide to Writing Integration Tests"
date:   2023-07-15 13:48:25 +0200
categories: Machine Learning Design
---

# Machine Learning System Testing: A Guide to Writing Integration Tests :space_invader:

Integration tests are crucial for verifying the functionality of complete machine learning pipelines. Minor adjustments in code, such as slight feature engineering or functional parameter modifications, can yield unexpected outputs. So, how can we effectively monitor these changes? This article will delve into the creation of integration tests tailored for machine learning systems :eyes:.

To begin, it's necessary to establish an ideal or expected system that serves as the reference point for results. These outcomes are retained and subsequently compared each time a modification is introduced to the system :lion:.

Determining how and what to compare depends on the specific objectives of the machine learning system. The optimal approach involves comparing the same metrics employed during the training and evaluation stages of the ML system :construction_worker_man:.

- **For data/feature engineering tasks**: Check if the features/vectors are computed as expected
- **For tasks involving classification**:
  Precision, recall, F-beta score, true positives (TP), false negatives (FN), true negatives (TN), false positives (FP)
- **For clustering tasks**:
  Number of clusters, purity
- **For regression tasks**:
  Mean squared error (MSE), mean absolute error (MAE)

Additionally, it is crucial to take into account `time` limitations, including factors like the `training time` and the `inference latency`.

# Setting up expected system :tiger:
Allow me to illustrate the procedure of establishing an integration test using an example.

The example has following objective and evaluation metrics:

  - **Objective**: Train clustering algorithm on the [iris dataset](https://gist.github.com/netj/8836201) and identify the cluster to which the provided sample belongs :hibiscus:.

  - **Evaluation Metrics**: Number of clusters and purity

Hence, we need to track the number of clusters and purity so that we can compare these metrics while testing the new changes. Now let us setup a simple training pipeline.

The following code defines a function `train()` which reads data from a CSV file named 'iris.csv' (located in a directory named 'dataset'). It extracts the features (sepal length, sepal width, petal length, petal width) and performs the following steps:

- Encodes the 'variety' column of the data using LabelEncoder, which assigns numerical labels to the categorical values.

- Applies K-Means clustering algorithm with a specified number of clusters (num_clusters). The resulting cluster labels are assigned to the data points.

- Computes a confusion matrix to analyze the agreement between the true labels (y_true) and the cluster labels obtained from K-Means.

- Calculates cluster purity for each cluster using the confusion matrix. Cluster purity measures the extent to which data points in a cluster belong to a single class.

- Collects the results, including the number of clusters and the average cluster purity, into a dictionary named results.

- Exports the results dictionary as a JSON file named 'results.json' (located in a directory named 'output/observed').

To being with,

- Download the the [iris.csv](https://gist.github.com/netj/8836201) and store it in the folder named `dataset`.
- Create two folders `output/expected` and `output/observed` which is the location to store the `expected` and `observed` output respectively.

Create a file named `train.py` and add the following code,

{% highlight python %}
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import json

def train(expected=True):
    # Load iris dataset
    data = pd.read_csv('dataset/iris.csv')
    X = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(data['variety'])

    # Perform K-Means clustering
    num_clusters = 3  # You can change this to the number of desired clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, cluster_labels)

    # Compute cluster purity
    cluster_purities = []
    for i in range(num_clusters):
        max_class_in_cluster = np.argmax(conf_matrix[:, i])
        cluster_purity = conf_matrix[max_class_in_cluster, i] / np.sum(conf_matrix[:, i])
        cluster_purities.append(cluster_purity)

    results = {
        "num_clusters": num_clusters,
        "purity": np.mean(cluster_purities)
    }
    print(results)
    type  = "expected" if expected else "observed"
    output_file = f"output/{type}/results.json"
    # Write the dictionary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)
    print(f"results exported to {output_file} as JSON")

if __name__=="__main__":
    train()
{% endhighlight %}

- Run `train.py`.
  {% highlight bash %}
  python train.py
  {% endhighlight %}
  You should now see the results exported into `output/expected` folder.
- Check `expected/results.json` for the results.
  {% highlight json %}
  {"num_clusters": 3, "purity": 0.9071873231465761}
  {% endhighlight %}

  **Validating** these outcomes is crucial as they serve as the ground truth upon which integration tests base their decisions.

This is the current desired structure of the directory tree.

{% highlight bash %}
├── dataset
│   └── iris.csv
├── output
│   ├── expected
│   │   └── results.json
│   └── observed
└── train.py
{% endhighlight %}

You have now setup the **expected system behaviour** :medal_military:.

# Setting up integration checker

The integration checker performs two essential tasks:

- Executes the training pipeline
- Compares the output of the expected system with the system being tested.

Create new file called `integration_checker.py`.

{% highlight python %}
from train import train
import json
import numpy as np

if __name__=="__main__":
    train()
    with open('output/expected/results.json') as json_data:
        expected = json.load(json_data)
    with open('output/observed/results.json') as json_data:
        observed = json.load(json_data)
    
    assert expected["num_clusters"] == observed["num_clusters"], "FAIL"
    assert np.isclose(expected["purity"], observed["purity"]), "FAIL"

    print("SUCCESS!")

{% endhighlight %}

This involves a comparison between the cluster count and cluster purity of the two systems. If either of these criteria is not met, an assertion error is triggered, accompanied by a `FAIL` message :x:.

Everything needed for setting up the testing system is now in place.

# Testing the new changes

In order to validate the modifications in `train.py`, we execute `integration-checker.py` without making any adjustments to `train.py`. Since the system being tested matches the expected system, the integration test should successfully pass :heavy_check_mark:.

{% highlight bash %}
python integration_checker.py
{'num_clusters': 3, 'purity': 0.9071873231465761}
results exported to output/observed/results.json as JSON
SUCCESS!
{% endhighlight %}

Next, we'll modify `train.py` by adjusting the number of clusters to 2 using `num_clusters = 2`, and then proceed to rerun the pipeline.

{% highlight bash %}
python integration_checker.py
{'num_clusters': 2, 'purity': 0.7294300719704337}
results exported to output/observed/results.json as JSON
Traceback (most recent call last):
  File "/Users/sourabhdattawad/Desktop/temp_test/integration_checker.py", line 11, in <module>
    assert expected["num_clusters"] == observed["num_clusters"], "FAIL"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: FAIL
{% endhighlight %}

Observing that the cluster count is suboptimal and the purity decreases to `0.72`, the test does not pass because the current system significantly diverges from the expected system :x:.

While the current approach might seem straightforward, it proves highly beneficial when assessing the validity of a complex large-scale system with numerous subtasks :smile:!

That concludes this post. Please reach out if you need further clarification or additional information :wave:.