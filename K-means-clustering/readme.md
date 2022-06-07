# K-means: <br/>
In classic k-means, we seek to minimize an Euclidean distance between the cluster center and the members of the cluster. The intuition behind this is that the radial distance from the cluster-center to the element location should "have sameness" or "be similar" for all elements of that cluster.<br/> <br/>

The algorithm is:
- Set number of clusters (aka cluster count).
- Initialize by randomly assigning points in the space to cluster indices.
- Repeat until converge.
    - For each point find the nearest cluster and assign point to cluster.
    - For each cluster, find the mean of member points and update center mean.
    - Error is norm of distance of clusters.

# K-means For Image Segmentation: <br/>
![Capture](https://user-images.githubusercontent.com/78277535/150841044-ff5ae7d5-ed30-420c-a4e0-22a1fa693b25.PNG)
![2](https://user-images.githubusercontent.com/78277535/150841202-60763a73-dd24-4d53-a488-ba012531d048.PNG)
![Capture](https://user-images.githubusercontent.com/78277535/150840044-85830e11-6a04-4aff-a24e-86fa9cc4e9b5.PNG)
