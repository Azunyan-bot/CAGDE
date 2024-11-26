import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc
import time

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

# ml_1m
#user_size,item_size=6040,3952
# ml_100k
user_size,item_size=943,1682
# Pinterest
# user_size,item_size=37501,9836
# citeulike
# user_size,item_size=5551,16981
# gowalla
#user_size, item_size = 29858, 40981
best_loss = 0
counter = 0
patience = 10
# torch.cuda.set_device(1)
result = []
# specicy the dataset, batch_size, and learning_rate.
dataset = 'ml_100k'
batch_size = 256
learning_rate = 2.0
adv_learning_rate = 0.01

# load the data

df_train = pd.read_csv(dataset + r'/train_sparse.csv')
df_test = pd.read_csv(dataset + r'/test_sparse.csv')

train_samples = 0
test_data = [[] for i in range(user_size)]
for row in df_train.itertuples():
    # train_data[row[1]].append(row[2])
    train_samples += 1
for row in df_test.itertuples():
    test_data[row[1]].append(row[2])

# interaction matrix
rate_matrix = torch.Tensor(np.load(dataset + r'/rate_sparse.npy')).cuda()


# user_size, item_size: the number of users and items
# beta: The hyper-parameter of the weighting function
# feature_type: (1) only use smoothed feautes (smoothed), (2) both smoothed and rough features (borh)
# drop_out: the ratio of drop out \in [0,1]
# latent_size: size of user/item embeddings
# reg: parameters controlling the regularization strength
class GDE(nn.Module):
    def __init__(self, user_size, item_size, beta=4.0, feature_type='smoothed', drop_out=0.2, latent_size=64, reg=0.01):
        super(GDE, self).__init__()
        self.user_embed = torch.nn.Embedding(user_size, latent_size)
        self.item_embed = torch.nn.Embedding(item_size, latent_size)

        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)


        self.beta = beta
        self.reg = reg
        self.drop_out = drop_out
        if drop_out != 0:
            self.m = torch.nn.Dropout(drop_out)

        if feature_type == 'smoothed':
            user_filter = self.weight_feature(
                torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_user_values.npy')).cuda())
            item_filter = self.weight_feature(
                torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_item_values.npy')).cuda())

            user_vector = torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_user_features.npy')).cuda()
            item_vector = torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_item_features.npy')).cuda()


        elif feature_type == 'both':

            user_filter = torch.cat([self.weight_feature(
                torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_user_values.npy')).cuda()) \
                                        , self.weight_feature(
                    torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_rough_user_values.npy')).cuda())])

            item_filter = torch.cat([self.weight_feature(
                torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_item_values.npy')).cuda()) \
                                        , self.weight_feature(
                    torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_rough_item_values.npy')).cuda())])

            user_vector = torch.cat(
                [torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_user_features.npy')).cuda(), \
                 torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_rough_user_features.npy')).cuda()], 1)

            item_vector = torch.cat(
                [torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_smooth_item_features.npy')).cuda(), \
                 torch.Tensor(np.load(r'./' + dataset + r'/' + dataset + '_rough_item_features.npy')).cuda()], 1)


        else:
            print('error')
            exit()

        self.L_u = (user_vector * user_filter).mm(user_vector.t())
        self.L_i = (item_vector * item_filter).mm(item_vector.t())

        del user_vector, item_vector, user_filter, item_filter
        gc.collect()
        torch.cuda.empty_cache()


    def weight_feature(self, value):
        return torch.exp(self.beta * value)

    def forward(self, user, pos_item, nega_item, epoch, loss_type='bpr'):

        user1 = copy.deepcopy(user)
        pos_item1 = copy.deepcopy(pos_item)
        nega_item1 = copy.deepcopy(nega_item)
        final_user1, final_pos1, final_nega1 = self.L_u[user1].mm(self.user_embed.weight), self.L_i[pos_item1].mm(self.item_embed.weight), self.L_i[nega_item1].mm(self.item_embed.weight)

        final_user, final_pos, final_nega = (self.m(self.L_u[user]) * (1 - self.drop_out)).mm(self.user_embed.weight), (self.m(self.L_i[pos_item]) * (1 - self.drop_out)).mm(self.item_embed.weight),(self.m(self.L_i[nega_item]) * (1 - self.drop_out)).mm(self.item_embed.weight)


        if loss_type == 'adaptive':

            res_nega = (final_user * final_nega).sum(1)
            nega_weight = (1 - (1 - res_nega.sigmoid().clamp(max=0.99)).log10()).detach()
            out = ((final_user * final_pos).sum(1) - nega_weight * res_nega).sigmoid()

        else:
            out = ((final_user * final_pos).sum(1) - (final_user * final_nega).sum(1)).sigmoid()

        reg_term = self.reg * (final_user ** 2 + final_pos ** 2 + final_nega ** 2).sum()

        cl_loss = clloss(final_user1, final_pos1, 0.1, 0.2)
        loss = (-torch.log(out).sum() + reg_term + 0.2 *cl_loss ) / batch_size
        if epoch not in range(100, 600):
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

    def predict_matrix(self):

        final_user = self.L_u.mm(self.user_embed.weight)
        final_item = self.L_i.mm(self.item_embed.weight)
        # mask the observed interactions
        return (final_user.mm(final_item.t())).sigmoid() - rate_matrix * 1000

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
        predict = self.predict_matrix()

        idcg_set = cal_idcg()
        for now_user in range(user_size):
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
        ndcg5, ndcg10, ndcg20, ndcg30, ndcg40, ndcg50, recall5, recall10, recall20, recall30, recall40, recall50 = round(ndcg5 / user_size, 4), round(ndcg10 / user_size, 4), round(ndcg20 / user_size, 4), round(ndcg30 / user_size, 4), round(ndcg40 / user_size, 4), round(ndcg50 / user_size, 4), round(recall5 / user_size, 4), round(
            recall10 / user_size, 4), round(recall20 / user_size, 4), round(recall30 / user_size, 4), round(recall40 / user_size, 4), round(recall50 / user_size, 4)
        print(ndcg5, ndcg10, ndcg20, ndcg30, ndcg40, ndcg50, recall5, recall10, recall20, recall30, recall40, recall50)
        result.append([ndcg5, ndcg10, ndcg20, ndcg30, ndcg40, ndcg50, recall5, recall10, recall20, recall30, recall40, recall50])



# Model training and test

model = GDE(user_size, item_size).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
advoptimizer = torch.optim.SGD(model.parameters(), lr=adv_learning_rate)
epoch = train_samples // batch_size

for i in range(500):
    total_loss = 0.0
    # start=time.time()
    for j in range(0, epoch):
        if i not in range(100, 600):
          optimizer.zero_grad()
          u = torch.LongTensor(np.random.randint(0, user_size, batch_size)).cuda()
          p = torch.multinomial(rate_matrix[u], 1, True).squeeze(1)
          nega = torch.multinomial(1 - rate_matrix[u], 1, True).squeeze(1)

          loss = model(u, p, nega, i)

          optimizer.step()

          total_loss += loss.item()
        else:
          advoptimizer.zero_grad()
          u = torch.LongTensor(np.random.randint(0, user_size, batch_size)).cuda()
          p = torch.multinomial(rate_matrix[u], 1, True).squeeze(1)
          nega = torch.multinomial(1 - rate_matrix[u], 1, True).squeeze(1)

          loss = model(u, p, nega, i)

          advoptimizer.step()

          total_loss += loss.item()


    # end=time.time()
    # print(end-start)
    print('epoch %d training loss:%f' % (i, total_loss / epoch))
    if (i + 1) % 10 == 0 and (i + 1) >= 90:
        model.test()
        if best_loss == 0:
            best_loss = result[-1][2]
            print ("best_loss =", best_loss)
        elif best_loss - result[-1][2] < 0:
            best_loss = result[-1][2]
            # reset counter if validation loss improves
            counter = 0
            print("best_loss =", best_loss,"counter = 0")
        elif best_loss - result[-1][2] > 0:
            counter += 1
            print(f"INFO: Early stopping counter {counter} of {patience}")
            if counter >= patience:
                print('INFO: Early stopping, best loss =', best_loss)
                break


#output = pd.DataFrame(result)
#output.to_csv(r'./gde.csv')