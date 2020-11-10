# Useful code for future sections


### Trainining/testing ###

# device = torch.device('cpu') # Can change to gpu if available.
# in_size = 7 # Number of days on which the prediction is based (features).
# H = 10 size of hidden layer
# out_size = 1 # Number of days to be predicted.
# learning_rate = 0.001 # step size
# batch_size = 32 # https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# num_epoch = 1 # 1 epoch means the network has seen the complete data set (Seen all batches).

# model = ANN(in_size,out_size).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) 


# # save model and import it.
# # Plot of predicted and oserved data.
