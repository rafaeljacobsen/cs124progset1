#!/usr/bin/env python3
import sys
import numpy as np
np.set_printoptions(suppress=True)

#graph option 1: generates an undirected graph between n vertices with each
#weight being uniformly distributed between 0 and 1
def generate_complete_graph(n,cutoff):
    adjacency={i: [] for i in range(n)}
    randoms=np.random.uniform(low=np.nextafter(0.0, 1.0),high=1.0, size=(n, n))
    ivals, jvals = np.where(np.triu(randoms < cutoff,k=1))
    for i, j in zip(ivals, jvals):
        adjacency[i].append([j, randoms[i, j]])
        adjacency[j].append([i, randoms[i, j]])
    return adjacency

#generates n vertices distributed at random within the unit hypercube
# and has the weights be the euclidian distance between them
def generate_unit_hypercube(n,d,cutoff):
    vertices=np.random.uniform(low=np.nextafter(0.0, 1.0),
                               high=1.0,
                               size=(n,d))

    adjacency={i: [] for i in range(n)}
    for i in range(n):
        distances = np.sqrt(np.sum((vertices[i]-vertices[i+1:])**2, axis=1))
        valids = np.where(distances < cutoff)[0]
        for k in valids:
            distance = distances[k]
            adjacency[i].append([i+1+k, distance])
            adjacency[i+1+k].append([i, distance])
    return adjacency

def generate_hypercube(n,cutoff):
    adjacency={i: [] for i in range(n)}
    k = int(np.log2(n))
    randoms = np.random.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=(n, k+1))

    exps=np.arange(k+1)
    for a in range(n):
        bs = a+2**exps
        bs=bs[bs<n]
        valids = np.where(randoms[a, :len(bs)] < cutoff)[0]
        for i in valids:
            adjacency[a].append([bs[i],randoms[a, i]])
            adjacency[bs[i]].append([a,randoms[a, i]])
    return adjacency

def generate_graph(n,d,cutoff):
    if d==0:
        return(generate_complete_graph(n,cutoff))
    if d==1:
        return(generate_hypercube(n,cutoff))
    if d==2 or d==3 or d==4:
        return(generate_unit_hypercube(n,d,cutoff))

#binary heap implementation
class BinaryHeap:
    def __init__(self):
        #values are stored in [vertex,weight] pairs
        self.values=[]
        self.locs={}
    #sorts the heap starting from the vertex bottom up
    def upheap(self,index):
        # if at the top of the heap, return
        while index != 0:
            #finds the parent index
            parent_index = (index-1)//2
            #if less than the parent index, then switch
            if self.values[index][1]<self.values[parent_index][1]:
                temp=self.values[index]
                self.values[index]=self.values[parent_index]
                self.values[parent_index]=temp
                self.locs[self.values[index][0]]=index
                self.locs[self.values[parent_index][0]]=parent_index
            #call again from the parent index
            index = parent_index
    #sorts the heap top down
    def downheap(self,index):
        while True:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            smallest = index
    
            if left_child_index < len(self.values) and self.values[left_child_index][1] < self.values[smallest][1]:
                smallest = left_child_index
    
            if right_child_index < len(self.values) and self.values[right_child_index][1] < self.values[smallest][1]:
                smallest = right_child_index
            if smallest == index:
                return
            temp=self.values[index]
            self.values[index]=self.values[smallest]
            self.values[smallest]=temp
            self.locs[self.values[index][0]]=index
            self.locs[self.values[smallest][0]]=smallest
            index = smallest
    #inserts a value and then sorts the heap
    def insert(self,vertex,value):
        self.values.append([vertex,value])
        self.locs[vertex]=len(self.values)-1
        self.upheap(len(self.values)-1)
    #gets rid of the root (smallest element)
    def deletemin(self):
        if len(self.values)==1:
            self.locs={}
            temp=self.values[0]
            self.values=[]
            return temp[0]
        min=self.values[0]
        self.values[0]=self.values[-1]
        self.locs[self.values[0][0]]=0
        remove=self.values.pop(-1)
        self.locs.pop(min[0])
        self.downheap(0)
        return(min[0])
    def decreasekey(self,vertex,value):
        self.values[self.locs[vertex]]=[vertex,value]
        self.upheap(self.locs[vertex])
    def checkvalid(self):
        for i in range(len(self.values)):
            if 2*i+1<len(self.values):
                if self.values[i][1]>self.values[2*i+1][1]:
                    return False
            if 2*i+2<len(self.values):
                if self.values[i][1]>self.values[2*i+2][1]:
                    return False
        return True

def find_cutoff(mult,a,b,n):
    return mult*a*n**(-b)
def primadj(adjacency):
    H = BinaryHeap() # initialize binary heap
    d = np.full(len(adjacency), np.inf) # set to infinity
    S = set() # set implementation, contains all vertices already connected
    d[0] = 0 # choose vertex s to be 0 (start with 0)
    H.insert(0,0) # insert into heap
    while len(H.values) != 0:
        u = H.deletemin()
        S.add(u)
        for pair in adjacency[u]:
            v=pair[0]
            # is an edge (nonzero), is not already in mst
            # if the weight is less than the d value stored, change
            if v not in S:
                if d[v] > pair[1]:
                    d[v] = pair[1]
                    if v in H.locs:
                        H.decreasekey(v,d[v])
                    else:
                        H.insert(v,d[v])
        assert(H.checkvalid())
    if np.sum(d)==np.inf:
        return []
    return d

def generategraphs(trials,n,d,cutoff):
    sums=[]
    for i in range(trials):
        E=generate_graph(n,d,cutoff)
        p=primadj(E)
        if len(p)==0:
            return generategraphs(trials,n,d,1.5*cutoff)
        sums.append(np.sum(p))
    return np.mean(sums)

def main():
    _, option, numpoints, numtrials, dimension = sys.argv
    dimension=int(dimension)
    numpoints=int(numpoints)
    numtrials=int(numtrials)
    if dimension == 0:
        cutoff=find_cutoff(1.5,2.1,0.8,numpoints)
    elif dimension== 1:
        cutoff=0.5
    elif dimension== 2:
        cutoff=find_cutoff(1.3,1.19,0.43,numpoints)
    elif dimension== 3:
        cutoff=find_cutoff(1.2,1.3,0.3,numpoints)
    elif dimension== 4:
        cutoff=find_cutoff(1.2,1.3,0.24,numpoints)
    else:
        print("error")
    average=generategraphs(numtrials,numpoints,dimension,cutoff)
    print(average,numpoints,numtrials,dimension)

if __name__ == "__main__":
    main()
