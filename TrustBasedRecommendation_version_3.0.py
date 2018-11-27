
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from sklearn.model_selection import train_test_split


# In[2]:


## creating file for taking outputs 
print("Opening file output.txt for writing outputs..........")
file = open("output.txt",'w')
file.write("Dataset Used is custom_trust and custom_rating\n")
file.write("Loading the rating and trust dataset.....\n")


# ### Reading The Rating and Trust Dataset

# In[3]:


#Loading the dataset
print("Loading the dataset.........")
df_trust=pd.read_csv("trust_data.txt",delim_whitespace=True,encoding="utf-8", skipinitialspace=True)

df_rating = pd.read_csv("ratings_data.txt")
df_rating = df_rating[(df_rating.userId<=100)&(df_rating.movieId<=4200)]
df_rating = df_rating.reset_index(drop=True)
file.write("Dataset Loaded Successfully!!!\n")
df_trust = df_trust[(df_trust.user1<=100)&(df_trust.user2<=100)]
df_trust = df_trust.reset_index(drop=True)
print("dataset loaded successfully!")
# test=df_rating[7903:]
# test
# df_trust

# temp=df_rating.sample(10).reset_index(drop=True)
# temp
# total_rows = df_rating.shape[0]
# total_rows


# In[4]:


# mov =df_rating.movieId.unique()
# # mov
# mov2=df_rating.sort_values(['movieId'], ascending=[True])
# mov2


# In[5]:


#as the partitioning of dataset is only done for rating not for trust so below line finds the overall users in the system
temp1 =df_rating.userId.unique()
temp2 = df_rating.userId.unique().shape[0]
total_users_for_trust=temp1[temp2-1]
file.write("Dividing The Rating Dataset into training and test.....\n")
# total_users_for_trust


# ### Dividing The Rating Dataset into training and test

# In[6]:


#dividing the dataset into training and test(80:20)

train=df_rating[0:3828]
test=df_rating[3828:]
test = test.reset_index(drop=True)
#df_rating is the training part of rating dataset and test is the testing part of the dataset
df_rating=train
# print("dataset divided successfully!!!")
file.write("completed!!!\n")


# ### Intializing the matrices for trust propagation,similarity and user pair distances

# In[60]:


# total_users calculates the total users in the training dataset 
#unique_user_list finds the unique users from the training dataset as there are redundant rows of the same user in the rating dataset

file.write("Intializing the matrices for trust propagation,similarity and user pair distances.................\n")
print("Intializing the matrices for trust propagation,similarity and user pair distances.................")
total_users = df_rating.userId.unique().shape[0]
unique_user_list = df_rating.userId.unique().tolist()

# following matrix i.e similarity is used for holding the pcc between a pair of users
# similarity = [[0 for x in range(total_users)] for y in range(total_users)]
similarity = np.zeros((total_users,total_users))

# following matrix i.e trust_in_users is used for holding the chain of trust between a pair of users
# trust_in_users = [[0 for x in range(total_users_for_trust)] for y in range(total_users_for_trust)]
trust_in_users = np.zeros((total_users_for_trust,total_users_for_trust))

# following matrix i.e user_pair_distance is used for holding the distance between a pair of users
#the formula used for calculating the distances between users is taken from novel 2d graph research paper
#distance of 9 tells that there is no way between pair of users to calculate distance so 9 denotes the infinity
# user_pair_distance = [[0 for x in range(total_users)] for y in range(total_users)]
user_pair_distance = np.zeros((total_users,total_users))

print("matrix intialization completed!!!")
file.write("completed!!!\n")

file.write("Construction of Novel 2D Graph started........\n")
print("-----------Construction of Novel 2D Graph started-----------------")


# In[73]:


# df_rating = pd.read_csv("custom_rating.csv")


#this function calculates similarity between the pair of user as needed using the pearson similarity coefficient
def calculateSimilarity(i,j,dataset1,dataset2):
    firstUser = dataset1[dataset1.userId==i]
    firstUser.columns = ['userId1','movieId','rating1']
    
    secondUser = dataset2[dataset2.userId==j]
    secondUser.columns = ['userId2','movieId','rating2']
    userRatings = pd.merge(firstUser,secondUser,on ='movieId')
    

    if(userRatings.shape[0]>1):
        
        
        firstUserRatingMean= firstUser["rating1"].mean()
        secondUserRatingMean= secondUser["rating2"].mean()


        firstUsertemp1=userRatings["rating1"]-firstUserRatingMean
        firstUsertemp1 =firstUsertemp1.fillna(0)
        firstUsertemp1sq=firstUsertemp1*firstUsertemp1
        firstUsertemp1sqsum=sum(firstUsertemp1sq)
        firstUsertemp1sqrt =math.sqrt(firstUsertemp1sqsum)


        secondUsertemp1=userRatings["rating2"]-secondUserRatingMean
        secondUsertemp1=secondUsertemp1.fillna(0)
        secondUsertemp1sq=secondUsertemp1*secondUsertemp1
        secondUsertemp1sqsum=sum(secondUsertemp1sq)
        secondUsertemp1sqrt =math.sqrt(secondUsertemp1sqsum)

        firstSecondtemp1=firstUsertemp1*secondUsertemp1
        firstSecondtemp1=firstSecondtemp1.fillna(0)
        firstSecondtemp1sum=sum(firstSecondtemp1)

        firstSecondtemp1sqrtMulti=firstUsertemp1sqrt*secondUsertemp1sqrt
        if(firstSecondtemp1sqrtMulti!=0):
            temp_pcc=firstSecondtemp1sum/firstSecondtemp1sqrtMulti
        else
            temp_pcc=0.00
    
        
        pcc=round(temp_pcc,2)
        
        
#         temp_pcc = np.corrcoef(userRatings['rating1'].tolist(),userRatings['rating2'].tolist())[0, 1]
    
    else:
        pcc = 0
    return pcc


# In[75]:


# simi=calculateSimilarity(1,5,df_rating,df_rating)
# simi
# # temp = df_rating['rating']-1
# # temp


# In[67]:


# temp1 = df_rating['rating'].head()-1
# # temp1
# temp2 = df_rating['rating']-2
# # temp2
# temp3 = df_rating['rating'].head()-temp2
# temp3=temp3.fillna(0)
# temp3


# ### Construction of Novel 2D Graph

# In[70]:


#module for calculating similarity currently pearson similarity is used

# # g is a graph that is used to show pcc similarity between users
# g = nx.DiGraph()
# g.add_nodes_from(unique_user_list)


# #this function calculates similarity between the pair of user as needed using the pearson similarity coefficient
# def calculateSimilarity(i,j,dataset1,dataset2):
#     firstUser = dataset1[dataset1.userId==i]
#     firstUser.columns = ['userId1','movieId','rating1']
#     secondUser = dataset2[dataset2.userId==j]
#     secondUser.columns = ['userId2','movieId','rating2']
#     userRatings = pd.merge(firstUser,secondUser,on ='movieId')
#     if(userRatings.shape[0]>1):
#         temp_pcc = np.corrcoef(userRatings['rating1'].tolist(),userRatings['rating2'].tolist())[0, 1]
#         pcc=round(temp_pcc,2)
#     else:
#         pcc = 0
#     return pcc

print("pcc similarity calculation started.........")
# this is used to calculate similarity between every pair of users
for i in range(0,total_users):
    print("pcc similarity calculation started for user id.........",i+1)
    for j in range(0,total_users):
        if(i==j):
            similarity[i][j]=round(1.000, 2)
        else:
            y= round(calculateSimilarity(i+1,j+1,df_rating,df_rating), 2)
            if(not np.isnan(y)):
                similarity[i][j]= y
print("pcc similarity completed!!!")
#      
# #for adding edges and weights to the graph g
# for i in range(0,total_users):
#     for j in range(i+1,total_users):
#         if(similarity[i][j]>0):
#             g.add_edge(i+1,j+1, weight=similarity[i][j])

            
# #this code shows the edge labels and final visualization of the graph 
# #here edge weights are pearson coorelation coefficient
# pos = nx.circular_layout(g)
# nx.draw(g,pos,with_labels=True)
# nx.draw_networkx_edge_labels(g,pos)
# plt.draw()
# plt.savefig("pcc_graph.png")
# plt.gcf().clear()
# # plt.show()  

# print(similarity)
print("printing similarity matrix................")
file.write("-----------------Printing the similarity matrix------------------------------\n")

for item in similarity:
    for sub_item in item:
        file.write("%s\t" % sub_item)
    file.write("\n")


file.write("\n\n\n")
file.write("----------------------------------------------------------------------------------------\n")

print("printing of similarity matrix completed!!!")
#########################################################################
#########################################################################
#########################################################################


#module for calculating chain of trust between users

# # t is a graph that is used to show trust between users
print("started making graph for trust propagation.....")
t = nx.DiGraph()
t.add_nodes_from(unique_user_list)

rows_in_trust = df_trust.shape[0]

for i in range(0,rows_in_trust):
    t.add_edge(df_trust.loc[i, 'user1'],df_trust.loc[i, 'user2'])
print("completed making graph for trust propagation!!!")

# pos = nx.circular_layout(t)
# nx.draw(t,pos,with_labels=True)
# # nx.draw_networkx_edge_labels(t,pos)
# plt.draw()
# plt.savefig("trust_chain_graph.png")
# plt.gcf().clear()
# # plt.show()

# here input x is based on the six degree theory
# if x is 2 then it means while calculating trust between two user
# a and b there can be atmost 2 node present between them
def calculateTrustChain(x):
    for i in range(1,total_users_for_trust+1):
        for j in range(1,total_users_for_trust+1):
            if(i!=j):
                print("calculating chain of trust between user id...",i,"...and user id...",j)
                try:
                    path_length = nx.shortest_path_length(t,source=i,target=j)
                    if(path_length <= x+1):
                        trust_in_users[i-1][j-1] = round(pow(path_length,-1),2)
                    else:
                        trust_in_users[i-1][j-1] = 0
                except nx.NetworkXNoPath:
                    trust_in_users[i-1][j-1] = 0
            else:
                trust_in_users[i-1][j-1] = 1
                
                    

print("calculating chain of trust with at max 2 hops.....")                    
file.write("\n---------calculating chain of trust using at max 2 hops between two users----------\n")      
calculateTrustChain(2)
print("chain of trust completed successfully!!!")
print("printing trust_in_users matrix to file.........")
# print(trust_in_users)

file.write("-----------------Printing the trust_in_users matrix------------------------------\n")

for item in trust_in_users:
    for sub_item in item:
        file.write("%s\t" % sub_item)
    file.write("\n")


file.write("\n\n\n")
file.write("----------------------------------------------------------------------------------------\n")

print("trust_in_users matrix printed successfully!!!")

#########################################################################
#########################################################################
#########################################################################


# #every edge of combined_graph is double weighted i.e having similarity and trust as its weights
# #and the users as its nodes
# combined_graph = nx.DiGraph()
# combined_graph.add_nodes_from(unique_user_list)

# #this loop adds edges between two nodes if there exist any edge
# for i in range(0,total_users):
#     for j in range(0,total_users):
#         if(i!=j):
#             if((trust_in_users[i][j]!=0) or (similarity[i][j]!=0)):
#                 combined_graph.add_edge(i+1,j+1, label='('+str(similarity[i][j])+','+str(trust_in_users[i][j])+')')
                
# #this code shows the combined graph in circular fashion
# pos = nx.circular_layout(combined_graph)
# nx.draw(combined_graph,pos,with_labels=True,node_size=100,font_size=10)
# nx.draw_networkx_edge_labels(combined_graph,pos,font_size=8)
# plt.draw()
# plt.savefig("pcc_and_trust_combined_graph.png")
# plt.gcf().clear()
# # plt.show()



#########################################################################
#########################################################################
#########################################################################



# This module is to make clusters out of given novel 2d graph

#these loops are calculating the distances between a pair of users
#the formula used for calculating distance is taken from novel 2d graph algorithm
#distance of 9 tells that there is no way between pair of users to calculate distance so it denotes infinity

print("started calculating user_pair_distance...........")
for i in range(0,total_users):
    for j in range(0,total_users):
        if(similarity[i][j]!=0 or trust_in_users[i][j]!=0):
            print("calculating user_pair_distance between user id...",i+1,"...and user id...",j+1)
            d_s = 1-similarity[i][j]
            d_t = 1-trust_in_users[i][j]
            d_s_2 = d_s**2
            d_t_2 = d_t**2
            temp = d_s_2+d_t_2
            distance = round(temp**0.5,2)
            user_pair_distance[i][j]=distance
        else:
            #here distance of 9 tells that there is no way between this pair of users so infinite distance
            user_pair_distance[i][j]=9
            
print("user_pair_distance calculated successfully!!!")

# print(user_pair_distance)
print("-----------Construction of Novel 2D Graph completed successfully!!!-----------------")
file.write("Construction of Novel 2D Graph Completed!!!!\n")
print("printing user_pair_distance to file....")
file.write("-----------------Printing the user_pair_distance matrix------------------------------\n")

for item in user_pair_distance:
    for sub_item in item:
        file.write("%s\t" % sub_item)
    file.write("\n")


file.write("\n\n\n")
print("user_pair_distance writing successfull!!!")
print("--------------------------------------------------------------------------------------------------")
file.write("----------------------------------------------------------------------------------------\n")


# ### Generating Feasible Partitioning

# In[62]:


# file.write("Started Generating Feasible Partitioning............\n")

def kmedoids(number_of_clusters):
    chosen_clusters=[]
    final_clusters=[]
    print("-----inside kemdoids chosing...",number_of_clusters,"......random cluster centers----")
    for i in range(0,number_of_clusters):
        x=random.randint(1,total_users)
        while(x in chosen_clusters):
            x=random.randint(1,total_users)
        chosen_clusters.append(x)
    for j in chosen_clusters:
        final_clusters.append([j])
    cont=True
    print("checking for corresponding cluster for each users(inside kmedoids).....!!! --Total_clustesr=",number_of_clusters)
    while(cont):
        for i in range(0,total_users):
            print("checking corresponding cluster for user id (inside kmedoids)....",i+1,"--Total_clusters=",number_of_clusters)
            if(not any(i+1 in sublist for sublist in final_clusters)):
                min_distance=10
                allocate_cluster=0
                for j in range(0,number_of_clusters):
                    temp_cluster=final_clusters[j][0]
                    if(min_distance>=user_pair_distance[i][temp_cluster-1]):
                        min_distance=user_pair_distance[i][temp_cluster-1]
                        allocate_cluster=j
                final_clusters[allocate_cluster].append(i+1)
        cont=False
        row_num=0
        modification_dict={}
        
        print("All users mapped to their corresponding clusters(inside kmedoids)!!!---Total_clusters : ",number_of_clusters)
        print("Checking each cluster for user which have minimum distance from all users in the same cluster(inside kmedoids).!-- Total_clusters :",number_of_clusters)
        
        for i in final_clusters:
            min_cluster_distance=99999
            for j in range(0,len(i)):
                total_sum=0
                avg=0
                for k in range(0,len(i)):
                    if(i[j]!=i[k]):
                        total_sum +=user_pair_distance[i[j]-1][i[k]-1]
                avg=total_sum/len(i)
                if(min_cluster_distance>avg):
                    min_cluster_distance=avg
                    min_cluster=i[j]
            if(min_cluster!=i[0]):
                cont=True
                modification_dict[row_num]=min_cluster                
            row_num+=1            
        if(cont):
            print("one of the cluster center has changed(inside kmedoids).........")
            print("Repeating the clustering with latest users as center(inside kmedoids).........")
            final_clusters=[]
            for z in chosen_clusters:
                    final_clusters.append([z])
            for key in modification_dict:
                    final_clusters[key][0]=modification_dict[key]
#     print(chosen_clusters)
#     print(final_clusters)
#     for i in range(0,len(final_clusters)):
#         cluster_centers.append(final_clusters[i][0])
    print("clustering into....",number_of_clusters,".....clusters completed(inside kmedoids)!!!--Total clusters formed:",number_of_clusters)
    return final_clusters    





#########################################################################
#########################################################################
#########################################################################





#this code checks the distance of each user in the test dataset with the 
#clusters and assigns nearest cluster to the user in the test dataset

def testUserDict(cluster_centers):
    print("checking test dataset user for their corresponding clusters...")
    testSetUser=test.userId.unique().tolist()
    targetClusterTestUserDict={}
    
#     print(cluster_centers)


    for i in range(0,len(testSetUser)):
        nearestClusterIndex=0
        minDistance=10
        print("checking test dataset user for their corresponding clusters...test Dataset row--",i+1)
        for j in range(0,len(cluster_centers)):

            pcc_simi=calculateSimilarity(testSetUser[i],cluster_centers[j][0],test,df_rating)
            temp1=testSetUser[i]-1
            temp2=cluster_centers[j][0]-1
            trust_simi=trust_in_users[temp1][temp2]
            if(pcc_simi!=0 or trust_simi!=0):
                d_s = 1-pcc_simi
                d_t = 1-trust_simi
                d_s_2 = d_s**2
                d_t_2 = d_t**2
                temp = d_s_2+d_t_2
                distance = round(temp**0.5,2)
            else:
                #here distance of 9 tells that there is no way between this pair of users so infinite distance
                distance=9
            if(minDistance>distance):
                minDistance=distance
                nearestClusterIndex=j
        targetClusterTestUserDict[testSetUser[i]]=nearestClusterIndex
    print("completed mapping of test dataset users into corresponding clusters successfully(inside testuserdict)!!!")
    return targetClusterTestUserDict


#########################################################################
#########################################################################
#########################################################################


#it calculates rating coverage for the system using the test dataset
def rating_coverage(clusters,targetClusterTestUserDict):
    count=0
    length=test.shape[0]
    for i in range(0,length):
        print("calculating rating coverage....for test Dataset row.....",i+1,"--for number of clusters:",len(clusters))
        userBelongsToCluster=targetClusterTestUserDict[test.iloc[i]['userId']]
        itemid=test.iloc[i]['movieId']
        for j in clusters[userBelongsToCluster]:
            rowNum=df_rating[(df_rating.userId==j)&(df_rating.movieId==itemid)].empty
            if(not rowNum):
                count+=1
                break;       
    rc=count/length
    rc=round(rc,2)
    print("rating coverage completed!!!--for number of clusters:",len(clusters))
    return rc*100




#########################################################################
#########################################################################
#########################################################################



#it checks and decides optimal number of clusters according to given rate of coverage
def feasible_partitioning(MARC):
    rc=100
    final_rc=0
    i=1
    partition1=[]
    partition2=[]
    while(rc>=MARC):
        i=i+1
        print("------------partioning the system into....",i,"........clusters------------------")
        partition2=kmedoids(i)
        targetClusterTestUserDict=testUserDict(partition2)
        final_rc=rc
        rc = rating_coverage(partition2,targetClusterTestUserDict)
        partition1=partition2
#     final_cluster=partition1
    
    print("\n\n\n######################################################################\n\n\n")
    print("Total Partitions/clusters formed after feasible partitioning is : ",len(partition1))
    print("\n\n\n######################################################################")
    
    return partition1,final_rc


# In[63]:


# # file.write("------------------------Started Generating Feasible Partitioning--------------------------------------------\n")

# abc=feasible_partitioning(70)
# file.write("Completed Feasible Partitioning!!!\n")

# file.write("-----------------------Printing the output of feasible partitioning at MARC=70--------------------\n")
# # abc

# for item in abc:
#     for sub_item in item:
#         file.write("%s\t" % sub_item)
#     file.write("\n")

# file.write("\n\n\n")

# file.write("---------------------------------------------------------------------------------------------------\n")


# In[ ]:


def check_cluster_performance_mae(a,b,c,cluster,final_clusters,jaccard_similarity_dict,rows_to_predict):
    
    real_rating=[]
    predicted_rating=[]
    length=rows_to_predict.shape[0]
    
    for i in range(0,length):
        
        print("calculating mae for cluster no.--",cluster+1,"---Total_cluster = ",len(final_clusters),"--checking rows_to_predict dataset row--",i+1)    
        
        real_rating.append(rows_to_predict.iloc[i]['rating'])
        temp_rating=predict_rating(final_clusters,rows_to_predict.iloc[i]['userId'],cluster,rows_to_predict.iloc[i]['movieId'],a,b,c,jaccard_similarity_dict,similarity)
        predicted_rating.append(temp_rating)
    
    diff = [abs(x - y) for x, y in zip(real_rating, predicted_rating)]
    diff_sum=sum(diff)
    length=len(diff)
    
    if(length != 0):
        mae=diff_sum/length
    else:
        mae=99
    return mae
    
    
    
    
    


# In[61]:


# temp = [[1,2,3],[4,5]]
# temp2=temp[1]
# temp3=df_rating[df_rating['userId'].isin(temp2)].reset_index(drop=True)
# temp3=temp3.sample(5)
# temp3

# temp = {2:{1:5,6:7}}
# # temp[2][7]
# temp={9:{}}
# temp2={}
# temp2[1]=5
# temp2[3]=6
# temp[9][2]=0
# temp[8]=temp2
# temp[8][1]
# temp[9][2]

# temp[5]=10
# temp[6]={}
# temp[6][1]=99
# temp
# jaccard_similarity_dict={}
# for i in temp[0]:
#         temp1={}
#         jaccard_similarity_dict[i]=temp1
# jaccard_similarity_dict[1][2]=5
# jaccard_similarity_dict[2][1]=6
# jaccard_similarity_dict


# ### Calculating optimal parameters i.e alpha, beta and gamma for each cluster

# In[64]:


#wuv is the convex combination of pearson,jaccard and trust
#it is calculated between user u and user v

def WUV(alpha,beta,gamma,u,v,jaccard_similarity_dict,testuser_pcc_similarity_dict):
    print("calculating wuv for (alpha,beta,gamma,u,v) :",alpha,beta,gamma,u,v)
#     pcc=calculateSimilarity(u,v,test,df_rating)
    pcc=testuser_pcc_similarity_dict[u-1][v-1]
    trust=trust_in_users[u-1][v-1]
#     jac=jaccard(trust_in_users[u-1],trust_in_users[v-1])
    jac=jaccard_similarity_dict[u][v]
    temp_total=alpha*pcc + beta*trust + gamma*jac
    total=round(temp_total,2)
    return total




#########################################################################
#########################################################################
#########################################################################


#jaccard is needed to calculate wuv 

def jaccard(a, b):
    print("calculating jaccard!!!....")
    intersection=0
    union=0
    for i in range(0,total_users):
        if(a[i]!=0 and b[i]!=0):
            intersection +=1
            union +=1
        elif(a[i]!=0):
            union +=1
        elif(b[i]!=0):
            union +=1
    temp_jac = float(intersection/union)
    jac=round(temp_jac,2)
    return jac



#########################################################################
#########################################################################
#########################################################################



#this code predicts the rating for a given user and movie by taking the trust and similarity into consideration

def predict_rating(final_clusters,userFromTest,userBelongsToCluster,itemid,alpha,beta,gamma,jaccard_similarity_dict,testuser_pcc_similarity_dict):
#     userBelongsToCluster=targetClusterTestUserDict[userFromTest]
    sum1=0
    sum2=0
    print("predicting rating for user id(test)",userFromTest,"movie id:",itemid,"--with (alpha,beta,gamma) as ",alpha,beta,gamma)
    for i in range(0,len(final_clusters[userBelongsToCluster])):
        clusterUser=final_clusters[userBelongsToCluster][i]
        if(clusterUser!=userFromTest):
            rowNum=df_rating[(df_rating.userId==clusterUser)&(df_rating.movieId==itemid)].empty
            if(not rowNum):
                clusterUserRating=df_rating[(df_rating.userId==clusterUser)&(df_rating.movieId==itemid)]['rating'].tolist()[0]
                temp_wuv=WUV(alpha,beta,gamma,userFromTest,clusterUser,jaccard_similarity_dict,testuser_pcc_similarity_dict)
                sum1+=temp_wuv*clusterUserRating
                sum2+=abs(temp_wuv)
    if(sum2==0):
        sum2=1
       
    prediction=sum1/sum2
    return round(prediction,2)


#########################################################################
#########################################################################
#########################################################################



#this code calculates the mean absoluter error of the system

def calc_MAE_system(alpha_beta_gamma_dict,targetClusterTestUserDict,final_clusters):
    real_rating=[]
    predicted_rating=[]
    length=test.shape[0]
    
    test_unique_user =test.userId.unique().tolist()
    
    jaccard_similarity_dict={}
    testuser_pcc_similarity_dict={}
    
    for i in test_unique_user:
        temp={}
        jaccard_similarity_dict[i]=temp
        testuser_pcc_similarity_dict[i-1]=temp
        for j in final_clusters[targetClusterTestUserDict[i]]:
            jac = jaccard(trust_in_users[i-1],trust_in_users[j-1])
            pcc=calculateSimilarity(i,j,test,df_rating)
            jaccard_similarity_dict[i][j]=jac 
            testuser_pcc_similarity_dict[i-1][j-1]=pcc
    
    
    
    
    for i in range(0,length):
        print("calculating mae for system---Total_cluster = ",len(final_clusters),"--checking test dataset row--",i+1)
        userBelongsToCluster=targetClusterTestUserDict[test.iloc[i]['userId']]
        real_rating.append(test.iloc[i]['rating'])
        temp_rating=predict_rating(final_clusters,test.iloc[i]['userId'],userBelongsToCluster,test.iloc[i]['movieId'],alpha_beta_gamma_dict[userBelongsToCluster]['alpha'],alpha_beta_gamma_dict[userBelongsToCluster]['beta'],alpha_beta_gamma_dict[userBelongsToCluster]['gamma'],jaccard_similarity_dict,testuser_pcc_similarity_dict)
        predicted_rating.append(temp_rating)
    
    diff = [abs(x - y) for x, y in zip(real_rating, predicted_rating)]
    diff_sum=sum(diff)
    length=len(diff)
    
    if(length != 0):
        mae=diff_sum/length
    else:
        mae=0
    return mae




#########################################################################
#########################################################################
#########################################################################


#this code checks which value of alpha,beta and gamma gives the minimum mean absolute error for a given cluster

def optimal_parameter(cluster,targetClusterTestUserDict,final_clusters):
    min=999
    temp_alpha=0
    temp_beta=0
    temp_gamma=1
    length=len(final_clusters[cluster])
    
    cluster_under_check = final_clusters[cluster]
    cluster_under_check_length=len(cluster_under_check)
    cluster_under_check_dataset = df_rating[df_rating['userId'].isin(cluster_under_check)].reset_index(drop=True)
    cluster_under_check_dataset_length=cluster_under_check_dataset.shape[0]
    average_row=cluster_under_check_dataset_length//cluster_under_check_length
    
    if(average_row>0):
        rows_to_predict=cluster_under_check_dataset.sample(average_row)
    else:
        rows_to_predict=cluster_under_check_dataset
    
    
    jaccard_similarity_dict={}
    
    for i in final_clusters[cluster]:
        temp={}
        jaccard_similarity_dict[i]=temp  
        
    for i in range(0,length):
        user1=final_clusters[cluster][i]
        for j in range(i+1,length):
            user2=final_clusters[cluster][j]
            jac= jaccard(trust_in_users[user1-1],trust_in_users[user2-1])
            jaccard_similarity_dict[user1][user2]=jac
            jaccard_similarity_dict[user2][user1]=jac
            
    
    for i in range(0,11,1):
        for j in range(0,11-i,1):
            print("-----calculating optimal parameter for cluster no....",cluster+1,"--with alpha =",temp_alpha,"--with beta=",temp_beta,"--")
            a=0.1*i
            a=round(a,2)
            b=0.1*j
            b=round(b,2)
            c=1-a-b
            c=round(c,2)
            print("calculating mae for cluster--",cluster+1)
#             mae = calc_MAE(a,b,c,cluster,targetClusterTestUserDict,final_clusters)

            mae = check_cluster_performance_mae(a,b,c,cluster,final_clusters,jaccard_similarity_dict,rows_to_predict)
            print("mae calculated for cluster-----",cluster+1,"---!!!")
            if(mae < min):
                min = mae
                temp_alpha = a
                temp_beta = b
                temp_gamma = c
    alpha=round(temp_alpha,2)
    beta=round(temp_beta,2)
    gamma=round(temp_gamma,2)
    print("optimal_parameter for cluster no.--",cluster+1,"-----------completed!!!")
    print("alpha=",alpha,"beta=",beta,"gamma=",gamma)
    print("-------------------------------------------")
    return alpha,beta,gamma


#########################################################################
#########################################################################
#########################################################################



#this code assigns optimal values of alpha,beta and gamma for each cluster

def ABGof_cluster():
    
    rc=70
    
    
    print("------------------------Started Generating Feasible Partitioning--------------------------------------------")
    file.write("------------------------Started Generating Feasible Partitioning--------------------------------------------\n")
    
    
    
    final_clusters,final_rc=feasible_partitioning(70)
    
    
#     print(final_clusters)
    print("feasible partitioning successfull!!!")
    file.write("Completed Feasible Partitioning!!!\n")

    file.write("------------------------Printing the output of feasible partitioning at MARC=70--------------------------\n")
    file.write("------------------------Printing final_clusters variable values for each cluster------------------------\n")
    print("printing output of feasible into file......")
    for item in final_clusters:
        file.write("[")
        for sub_item in item:
            file.write("%s," % sub_item)
        file.write("]\n")
    
    file.write("\n\n\n")
    print("writing completed successfully!!!")
    file.write("----------------------------------------------------------------------------------------------------------\n")
    print("calculating testUserDict using final clusters.........")
    targetClusterTestUserDict=testUserDict(final_clusters)
    length=len(final_clusters)
    final_alpha=[]
    final_beta=[]
    final_gamma=[]
    alpha_beta_gamma_dict={}
    for i in range(0,length):
        alpha_beta_gamma_dict[i]={}
        print("-------------------------calculating optimal parameter for cluster no....",i+1)
        tempa,tempb,tempc=optimal_parameter(i,targetClusterTestUserDict,final_clusters)
        alpha_beta_gamma_dict[i]['alpha']=tempa
        alpha_beta_gamma_dict[i]['beta']=tempb
        alpha_beta_gamma_dict[i]['gamma']=tempc
        
        final_alpha.append(tempa)
        final_beta.append(tempb)
        final_gamma.append(tempc)
    print("optimal parameter calculated successfully!!!")
    file.write("Completed Calculation!!!\n")
    
    file.write("------------------------Printing final_alpha variable values for each cluster------------------------\n")
    print("printing final_alpha to file....")
    for sub_item in final_alpha:
        file.write("%s\t" % sub_item)
        
    file.write("\n\n\n")
    print("completed!!!")
    file.write("----------------------------------------------------------------------------------------------------------\n")
    
    file.write("------------------------Printing final_beta variable values for each cluster------------------------\n")
    print("printing final_beta to file....")
    for sub_item in final_beta:
        file.write("%s\t" % sub_item)
    
    file.write("\n\n\n")
    print("completed!!!")
    file.write("----------------------------------------------------------------------------------------------------------\n")
    
    
    file.write("------------------------Printing final_gamma variable values for each cluster------------------------\n")
    print("printing final_gamma to file....")
    for sub_item in final_gamma:
        file.write("%s\t" % sub_item)
    print("completed!!!")
    file.write("\n\n\n")
    
    print("Calculating mean absolute error for the complete system!!!!!!!!!!!!!!!!!!!!")
    
    
    mae_of_system=calc_MAE_system(alpha_beta_gamma_dict,targetClusterTestUserDict,final_clusters)
    print("\n\n\n ----------Rating Coverage of system -----------: ",final_rc)
    
    
    file.write("\n\n\n ----------Rating Coverage of system -----------: ")
    file.write("%s\t" % final_rc)
    
    file.write("\n---------------------------------------\n\n")
    print("\n\n\n")
    
    
    print("\n\n\n ----------mae_of_system -----------: ",mae_of_system)
    
    
    file.write("\n\n\n ----------mae_of_system -----------: ")
    file.write("%s\t" % mae_of_system)
    
    file.write("\n---------------------------------------\n\n")
    print("\n\n\n")
    
    precision=1-(mae_of_system/4)
    
    print("\n\n\n ----------Precision of system -----------: ",precision)
    file.write("\n\n\n ----------Precision of system  -----------: ")
    file.write("%s\t" % precision)
    
    file.write("\n---------------------------------------\n\n")
    print("\n\n\n")
    
    f1_measure=(2*precision*final_rc)/(precision+final_rc)
    f1_measure=round(f1_measure,2)
    
    
    
    print("\n\n\n ----------F1 measure of system -----------: ",f1_measure)
    file.write("\n\n\n ----------F1 measure of system  -----------: ")
    file.write("%s\t" % f1_measure)
    
    file.write("\n---------------------------------------\n\n")
    print("\n\n\n")
    
    
    
    
    
    
    file.write("----------------------------------End Of Output File------------------------------------------------------------------------\n")
    print("--------------------------------End---------------------------------------")
    
#     print(final_alpha)
#     print(final_beta)
#     print(final_gamma)


    
    

#########################################################################
#########################################################################
#########################################################################


# In[70]:


# test_unique_user =test.userId.unique().tolist()
# test_unique_user


# In[65]:


file.write("Started Calculating optimal parameters i.e alpha, beta and gamma for each cluster............\n")
print("Started Calculating optimal parameters i.e alpha, beta and gamma for each cluster............")
ABGof_cluster()


# In[66]:


print("closing ouput.txt file............!!!")
file.close()

