Execute the project Python way:

1. In any windows folder open windows powershell (shift+right click)
2. Type "python -m venv env" - To create the python environment
3. Type ".\env\Scripts\activate" - To start the environment
4. Type "pip install -r requirements.txt" - To install the required packages for the project
5. Type "python basic_ml.py" 
6. Type "python api.py"
7. Test the REST API with FastAPI swagger UI http://127.0.0.1:8000


Execute the project Docker Way:

1. Type "docker build -t <image_name> <space> .(dot)
2. Type "docker run -p 8000:8000 --name <container_name> <image_name>
3. Test the REST API with FastAPI swagger UI http://127.0.0.1:8000

For the above steps it is essential that Docker daemon is installed into the system.

To stop and delete the entire docker setup:
 1. docker stop <container_name>
 2. docker rm <container_name>
 3. docker rmi <image_name>

Stop all the containers of this image:
    docker ps -q --filter ancestor<=<image_name> | xargs -r docker stop

Remove all the containers of this particular iamge:
    docker ps -aq --filter ancestor=<image_name> | xargs -r docker rm -f   

Start the kubernets Service to access the app through Loadbalancer of kubernets pods :

 kubectl port-forward service/nginx 8000:80 --address=0.0.0.0 --namespace my-namespace

kubectl delete pod -l app=houseprice -n dev

Point Docker CLI to Minikube’s Docker daemon:
        eval $(minikube docker-env)

Kubernets:
 1. First start the minikube : minikube start
 2. Redirect the docker to the minikube using : eval $(minikube docker-env)
 3. Build the docker image that will be used by the kubernetes:
     docker build -t apiimg .
 4. Create Kubernetes namespace:
     kubectl create namespace dev
 5. Create Kubernetes deployment pods:
     kubectl apply -f deployment.yaml -n dev
 6. Create Service of Kubernetes to access the pods:
     kubectl apply -f service.yaml -n dev
 7. Since this is the local docker image it needs to be forwarded to the port:
      kubectl port-forward service/houseprice 8000:8000 --address=0.0.0.0 --namespace dev    
