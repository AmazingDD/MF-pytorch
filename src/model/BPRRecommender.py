'''
@Author: Yu Di
@Date: 2019-08-20 10:52:03
@LastEditors: Yudi
@LastEditTime: 2019-08-20 13:51:41
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: BPR MF
'''
import torch

class BPR(torch.nn.Module):
    def __init__(self, params):
        super(BPR, self).__init__()
        user_num = params['user_num']
        item_num = params['item_num']
        factor_num = params['factor_num']

        self.embed_user = torch.nn.Embedding(user_num, factor_num)
        self.embed_item = torch.nn.Embedding(item_num, factor_num)
        torch.nn.init.normal_(self.embed_user.weight, std=0.01)
        torch.nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        pred_i = (user * item_i).sum(dim=-1)
        pred_j = (user * item_j).sum(dim=-1)
        return pred_i, pred_j

# ######################
# parser = argparse.ArgumentParser()
# parser.add_argument("--lr", 
# 	type=float, 
# 	default=0.01, 
# 	help="learning rate")
# parser.add_argument("--lamda", 
# 	type=float, 
# 	default=0.001, 
# 	help="model regularization rate")
# parser.add_argument("--batch_size", 
# 	type=int, 
# 	default=4096, 
# 	help="batch size for training")
# parser.add_argument("--epochs", 
# 	type=int,
# 	default=50,  
# 	help="training epoches")
# parser.add_argument("--top_k", 
# 	type=int, 
# 	default=10, 
# 	help="compute metrics@top_k")
# parser.add_argument("--factor_num", 
# 	type=int,
# 	default=32, 
# 	help="predictive factors numbers in the model")
# parser.add_argument("--num_ng", 
# 	type=int,
# 	default=4, 
# 	help="sample negative items for training")
# parser.add_argument("--test_num_ng", 
# 	type=int,
# 	default=99, 
# 	help="sample part of negative items for testing")
# parser.add_argument("--out", 
# 	default=True,
# 	help="save model or not")
# parser.add_argument("--gpu", 
# 	type=str,
# 	default="0",  
# 	help="gpu card ID")
# args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# cudnn.benchmark = True


# ############################## PREPARE DATASET ##########################
# train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()

# # construct the train and test datasets
# train_dataset = data_utils.BPRData(
# 		train_data, item_num, train_mat, args.num_ng, True)
# test_dataset = data_utils.BPRData(
# 		test_data, item_num, train_mat, 0, False)
# train_loader = data.DataLoader(train_dataset,
# 		batch_size=args.batch_size, shuffle=True, num_workers=4)
# test_loader = data.DataLoader(test_dataset,
# 		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

# ########################### CREATE MODEL #################################
# model = model.BPR(user_num, item_num, args.factor_num)
# model.cuda()

# optimizer = optim.SGD(
# 			model.parameters(), lr=args.lr, weight_decay=args.lamda)
# # writer = SummaryWriter() # for visualization

# ########################### TRAINING #####################################
# count, best_hr = 0, 0
# for epoch in range(args.epochs):
# 	model.train() 
# 	start_time = time.time()
# 	train_loader.dataset.ng_sample()

# 	for user, item_i, item_j in train_loader:
# 		user = user.cuda()
# 		item_i = item_i.cuda()
# 		item_j = item_j.cuda()

# 		model.zero_grad()
# 		prediction_i, prediction_j = model(user, item_i, item_j)
# 		loss = - (prediction_i - prediction_j).sigmoid().log().sum()
# 		loss.backward()
# 		optimizer.step()
# 		# writer.add_scalar('data/loss', loss.item(), count)
# 		count += 1

# 	model.eval()
# 	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

# 	elapsed_time = time.time() - start_time
# 	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
# 			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
# 	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

# 	if HR > best_hr:
# 		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
# 		if args.out:
# 			if not os.path.exists(config.model_path):
# 				os.mkdir(config.model_path)
# 			torch.save(model, '{}BPR.pt'.format(config.model_path))

# print("End. Best epoch {:03d}: HR = {:.3f}, \
# 	NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))