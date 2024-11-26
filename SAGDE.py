import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy
import pandas as pd
import torch.nn.functional as F
import time
import gc

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
set_seed(2024)

def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    Clloss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(Clloss)

def clloss(user, pos, rate, temperature):
    embedding1=[]
    embedding2=[]
    all_embeddings = torch.cat([user, pos], 0)
    random_noise1 = torch.rand_like(all_embeddings).cuda()
    random_noise2 = torch.rand_like(all_embeddings).cuda()
    all_embedding1 = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise1, dim=-1) * rate
    all_embedding2 = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise2, dim=-1) * rate
    embedding1.append(all_embedding1)
    embedding1 = torch.stack(embedding1, dim=1)
    embedding1 = torch.mean(embedding1, dim=1)
    embedding2.append(all_embedding2)
    embedding2 = torch.stack(embedding2, dim=1)
    embedding2 = torch.mean(embedding2, dim=1)
    user_view1, pos_view1 = torch.split(embedding1, [256, 256])
    user_view2, pos_view2 = torch.split(embedding2, [256, 256])
    user_cl_loss = InfoNCE(user_view1, user_view2, temperature)
    item_cl_loss = InfoNCE(pos_view1, pos_view2, temperature)
    return user_cl_loss + item_cl_loss

#gowalla
#user,item=29858,40981
#yelp
#user,item=25677,25815
#ml-1m
user,item=6040,3952

result=[]
dataset='ml_1m'
learning_rate=7.5
batch_size=256


df_train=pd.read_csv(dataset+ r'/train_sparse.csv')
df_test=pd.read_csv(dataset+ r'/test_sparse.csv')

#load the train/test data
#load the data
train_samples=0
#train_data=[[] for i in range(user)]
test_data=[[] for i in range(user)]
for row in df_train.itertuples():
	#train_data[row[1]].append(row[2])
	train_samples+=1
for row in df_test.itertuples():
	test_data[row[1]].append(row[2])
##########################################
#interaction matrix
rate_matrix=torch.Tensor(np.load(dataset+ r'/rate_sparse.npy')).cuda()


class SGDE(nn.Module):
	def __init__(self, user_size, item_size, beta=4.0, req_vec=60,std=0.02, latent_size=64, reg=0.01):
		super(SGDE, self).__init__()

		self.latent_size=latent_size
		self.std=std
		self.reg=reg
		self.beta=beta
		self.user_size=user_size
		self.item_size=item_size

		svd_filter=self.weight_func(torch.Tensor(np.load(dataset+ r'/svd_value.npy')[:req_vec]).cuda())
		self.user_vector=(torch.Tensor(np.load(dataset+ r'/svd_u.npy')[:,:req_vec])).cuda()*svd_filter
		self.item_vector=(torch.Tensor(np.load(dataset+ r'/svd_v.npy')[:,:req_vec])).cuda()*svd_filter
		self.FS=Variable(torch.nn.init.uniform_(torch.randn(req_vec,latent_size),-np.sqrt(6. / (req_vec+latent_size) ) ,np.sqrt(6. / (req_vec+latent_size) )).cuda(),requires_grad=True)

	def weight_func(self,sig):
		return torch.exp(self.beta*sig)



	def predict(self):

		final_user=self.user_vector.mm(self.FS)
		final_item=self.item_vector.mm(self.FS)
		return (final_user.mm(final_item.t())).sigmoid()-rate_matrix*1000


	def forward(self,u,p,n,epoch):

		final_user,final_pos,final_nega=torch.normal(self.user_vector[u],std=self.std).mm(self.FS),torch.normal(self.item_vector[p],std=self.std).mm(self.FS),torch.normal(self.item_vector[n],std=self.std).mm(self.FS)
		u1=copy.deepcopy(u)
		p1=copy.deepcopy(p)
		final_user1,final_pos1=torch.normal(self.user_vector[u1],std=self.std).mm(self.FS),torch.normal(self.item_vector[p1],std=self.std).mm(self.FS)
		out=((final_user*final_pos).sum(1)-(final_user*final_nega).sum(1)).sigmoid()
		regu_term=self.reg*(final_user**2+final_pos**2+final_nega**2).sum()
		cl_loss = clloss(final_user1, final_pos1, 0.1, 0.2)

		loss = (-torch.log(out).sum()+regu_term+ 0.2 *cl_loss)/batch_size
		if epoch not in range(10, 13):
			loss.backward()
			return loss
		else:
			final_user.retain_grad()
			final_pos.retain_grad()
			final_nega.retain_grad()
			loss.backward(retain_graph=True)
			grad_u = final_user.grad
			grad_p = final_pos.grad
			grad_n = final_nega.grad
			if grad_u is not None:
				delta_u = nn.functional.normalize(grad_u,p=2, eps=0.5)
			else:
				delta_u = torch.rand(final_user.size())
			if grad_p is not None:
				delta_p = nn.functional.normalize(grad_p,p=2, eps=0.5)
			else:
				delta_p = torch.rand(final_pos.size())
			if grad_n is not None:
				delta_n = nn.functional.normalize(grad_n,p=2, eps=0.5)
			else:
				delta_n = torch.rand(final_nega.size())
			adv_user = final_user+delta_u.to('cuda')
			adv_pos = final_pos+delta_p.to('cuda')
			adv_n = final_nega+delta_n.to('cuda')

			adv = ((adv_user * adv_pos).sum(1) - (adv_user * adv_n).sum(1)).sigmoid()

			adv_loss = -0.5*torch.log(adv).sum()/batch_size + loss
			adv_loss.backward()

			return adv_loss


	def test(self):
        # calculate idcg@k(k={1,...,20})
		def cal_idcg(k=50):
			idcg_set = [0]
			scores = 0.0
			for i in range(1, k + 1):
				scores += 1 / np.log2(1 + i)
				idcg_set.append(scores)

			return idcg_set

		def cal_score(topn, now_user, trunc=50):
			dcg5, dcg10, dcg20, dcg30, dcg40, dcg50, hit5, hit10, hit20, hit30, hit40, hit50 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
			for k in range(trunc):
				max_item = topn[k]
				if test_data[now_user].count(max_item) != 0:
					if k <= 5:
						dcg5 += 1 / np.log2(2 + k)
						hit5 += 1
					if k <= 10:
						dcg10 += 1 / np.log2(2 + k)
						hit10 += 1
					if k <= 20:
						dcg20 += 1 / np.log2(2 + k)
						hit20 += 1
					if k <= 30:
						dcg30 += 1 / np.log2(2 + k)
						hit30 += 1
					if k <= 40:
						dcg40 += 1 / np.log2(2 + k)
						hit40 += 1
					dcg50 += 1 / np.log2(2 + k)
					hit50 += 1

			return dcg5, dcg10, dcg20, dcg30, dcg40, dcg50, hit5, hit10, hit20, hit30, hit40, hit50

        # accuracy on test data
		ndcg5, ndcg10, ndcg20, ndcg30, ndcg40, ndcg50, recall5, recall10, recall20, recall30, recall40, recall50 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        #final_user = self.L_u.mm(self.user_embed.weight)
		predict = self.predict()

		idcg_set = cal_idcg()
		for now_user in range(user):
			test_lens = len(test_data[now_user])
			all5 = 5 if (test_lens > 5) else test_lens
			all10 = 10 if (test_lens > 10) else test_lens
			all20 = 20 if (test_lens > 20) else test_lens
			all30 = 30 if (test_lens > 30) else test_lens
			all40 = 40 if (test_lens > 40) else test_lens
			all50 = 50 if (test_lens > 50) else test_lens
			topn = predict[now_user].topk(50)[1]
			dcg5, dcg10, dcg20, dcg30, dcg40, dcg50, hit5, hit10, hit20, hit30, hit40, hit50 = cal_score(topn, now_user)
			ndcg5 += (dcg5 / idcg_set[all5])
			ndcg10 += (dcg10 / idcg_set[all10])
			ndcg20 += (dcg20 / idcg_set[all20])
			ndcg30 += (dcg30 / idcg_set[all30])
			ndcg40 += (dcg40 / idcg_set[all40])
			ndcg50 += (dcg50 / idcg_set[all50])
			recall5 += (hit5 / all5)
			recall10 += (hit10 / all10)
			recall20 += (hit20 / all20)
			recall30 += (hit30 / all30)
			recall40 += (hit40 / all40)
			recall50 += (hit50 / all50)
		ndcg5, ndcg10, ndcg20, ndcg30, ndcg40, ndcg50, recall5, recall10, recall20, recall30, recall40, recall50 = round(ndcg5 / user, 4), round(ndcg10 / user, 4), round(ndcg20 / user, 4), round(ndcg30 / user, 4), round(ndcg40 / user, 4), round(ndcg50 / user, 4), round(recall5 / user, 4), round(recall10 / user, 4), round(recall20 / user, 4), round(recall30 / user, 4), round(recall40 / user, 4), round(recall50 / user, 4)
		print(ndcg5, ndcg10, ndcg20, ndcg30, ndcg40, ndcg50, recall5, recall10, recall20, recall30, recall40, recall50)
		result.append([ndcg5, ndcg10, ndcg20, ndcg30, ndcg40, ndcg50, recall5, recall10, recall20, recall30, recall40, recall50])



#Model training and test

model = SGDE(user, item)

epoch=train_samples//batch_size

for i in range(13):
	total_loss,loss=0.0,0
	#start=time.time()
	for j in range(0,epoch):
#		if i not in range(90, 200):
		u=np.random.randint(0,user,batch_size)
		p=torch.multinomial(rate_matrix[u],1,True).squeeze(1)
		nega=torch.multinomial(1-rate_matrix[u],1,True).squeeze(1)
		
		loss=model(u,p,nega,i)
		
		with torch.no_grad():
			model.FS-=learning_rate*model.FS.grad
			model.FS.grad.zero_()
		total_loss+=loss.item()
#		else:
#      		u=np.random.randint(0,user,batch_size)
#			p=torch.multinomial(rate_matrix[u],1,True).squeeze(1)
#			nega=torch.multinomial(1-rate_matrix[u],1,True).squeeze(1)
#		
#			loss=model(u,p,nega,i)
#		
#			with torch.no_grad():
#				model.FS-=advlearning_rate*model.FS.grad
#				model.FS.grad.zero_()
#			total_loss+=loss.item()

	#end=time.time()
	#print(end-start)
	
	print('epoch %d training loss:%f' %(i,total_loss/epoch))
	if (i+1)%1==0 and (i+1)>=5:
		model.test()




output=pd.DataFrame(result)
output.to_csv(r'./rsgde.csv')

