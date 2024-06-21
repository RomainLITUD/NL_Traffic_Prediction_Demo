from explainabletf_utils.baselayer import *

class DGCN(nn.Module):
    def __init__(self, para, A, B, return_interpret=False):
        super(DGCN, self).__init__()
        self.N = para['nb_node']
        self.F = para['dim_feature']
        self.A = A
        self.B = B
        self.T = para['horizon']
        self.interpret = return_interpret
        
        self.encoder = Encoder(self.N, self.F, self.A, self.B)
        self.decoder = Decoder(self.N, self.F, self.A, self.B, self.T)

    def forward(self, x):
        h = self.encoder(x[:,:-self.T-1])
        prediction, mask1, mask2, demand = self.decoder(x[:,-self.T-1:-1], h)

        # h = self.encoder(x[:,:-1])
        # prediction, mask1, mask2, demand = self.decoder(x[:,-1], h)
        #print(mask1.size())
        if self.interpret:
            return prediction, mask1, mask2, demand
        return prediction