import torch.nn as nn
import torch
class Embedding(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,padding_value,max_seq,model_type="GRU"):
        super(Embedding, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.padding_value=padding_value
        self.max_seq=max_seq

        if model_type=="GRU":
            self.rnn=nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)
        else:
            self.rnn=nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.linear=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.sigmoid=nn.Sigmoid()
        self.reset()

    def reset(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self,x,t):
    #x:input_data [b,s,d]
    #t:time_length [b]
    #
        x_packed=nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=t,
            batch_first=True,
            enforce_sorted=False
        )
    
        H_o,H_t=self.rnn(x_packed)
        #print(t)
        H_o,T=nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq
        )   
    
        logits = self.linear(H_o)
        H = self.sigmoid(logits)
    #シーケンス長がバラバラの場合:難しいので今は実装なし。
        """mask=[]
        for i in t:
        mask.append(torch.cat([torch.ones(i,10),torch.zeros(max_seq-i,10)]))
        mask=torch.stack(mask)
        H=mask*H"""
    
        return H

class Supervisor(nn.Module):
    def __init__(self,hidden_dim,num_layers,padding_value,max_seq,model_type="GRU"):
        super(Supervisor, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.padding_value=padding_value
        self.max_seq=max_seq

        if model_type=="GRU":
            self.rnn=nn.GRU(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)
        else:
            self.rnn=nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.linear=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.sigmoid=nn.Sigmoid()
        self.reset()

    def reset(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self,H,t):
        #x:input_data [b,s,d]
        #t:time_length [b]
        #out:next_prediction_data [t-3,t-2,t-1,t,t+1]
        H_packed=nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=t,
            batch_first=True,
            enforce_sorted=False
        )
    
        H_o,H_t=self.rnn(H_packed)
        #print(H_o)
        #print(t)
        H_o,T=nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq
        )   
        
        logits = self.linear(H_o)
        H_ = self.sigmoid(logits)
        #シーケンス長がバラバラの場合:難しいので今は実装なし。
        """mask=[]
        print(t)
        for i in t:
            mask.append(torch.cat([torch.ones(i-1,10),torch.zeros(max_seq-i+1,10)]))
            mask=torch.stack(mask)

        H_=mask*H_"""
        
        return H_

class Recovery(nn.Module):
    def __init__(self,hidden_dim,output_dim,num_layers,padding_value,max_seq,model_type="GRU"):
        super(Recovery, self).__init__()
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        self.padding_value=padding_value
        self.max_seq=max_seq

        if model_type=="GRU":
            self.rnn=nn.GRU(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)
        else:
            self.rnn=nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)
        
        self.linear=nn.Linear(self.hidden_dim,self.output_dim)
        self.reset()

    def reset(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self,H,t):
        #x:input_data [b,s,d]
        #t:time_length [b]
        #
        H_packed=nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=t,
            batch_first=True,
            enforce_sorted=False
        )
        
        x_o,x_t=self.rnn(H_packed)
        #print(H_o)
        #print(t)
        x_o,T=nn.utils.rnn.pad_packed_sequence(
            sequence=x_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq
        )   
        
        x_ = self.linear(x_o)
        
        #シーケンス長がバラバラの場合:難しいので今は実装なし。
        """mask=[]
        print(t)
        for i in t:
        mask.append(torch.cat([torch.ones(i-1,10),torch.zeros(max_seq-i+1,10)]))
        mask=torch.stack(mask)

        H_=mask*H_"""
        
        return x_

class Generator(nn.Module):
    def __init__(self,latent_dim,hidden_dim,num_layers,padding_value,max_seq,model_type="GRU"):
        super(Generator, self).__init__()
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.padding_value=padding_value
        self.max_seq=max_seq

        if model_type=="GRU":
            self.rnn=nn.GRU(input_size=self.latent_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=self.num_layers,
                                    batch_first=True)
        else:
            self.rnn=nn.LSTM(input_size=self.latent_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=self.num_layers,
                                    batch_first=True)
        
        self.linear=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.sigmoid=nn.Sigmoid()
        self.reset()
    
    def reset(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self,z,t):
        #x:input_data [b,s,d]
        #t:time_length [b]
        #
        z_packed=nn.utils.rnn.pack_padded_sequence(
            input=z,
            lengths=t,
            batch_first=True,
            enforce_sorted=False
        )
        
        H_o,H_t=self.rnn(z_packed)
        
        H_o,T=nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq
        )   
        #print(H_o)
        logits = self.linear(H_o)
        H = self.sigmoid(logits)
        #シーケンス長がバラバラの場合:難しいので今は実装なし。
        """mask=[]
        for i in t:
        mask.append(torch.cat([torch.ones(i,10),torch.zeros(max_seq-i,10)]))
        mask=torch.stack(mask)
        H=mask*H"""
        
        return H

class Discriminator(nn.Module):
    def __init__(self,hidden_dim,num_layers,padding_value,max_seq,model_type="GRU"):
        super(Discriminator, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.padding_value=padding_value
        self.max_seq=max_seq

        if model_type=="GRU":
            self.rnn=nn.GRU(input_size=self.hidden_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True)
        else:
            self.rnn=nn.LSTM(input_size=self.hidden_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True)
        
        self.linear=nn.Linear(self.hidden_dim,1)
        self.sigmoid=nn.Sigmoid()
        self.reset()
    
    def reset(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self,H,t):
        #x:input_data [b,s,d]
        #t:time_length [b]
        #
        H_packed=nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=t,
            batch_first=True,
            enforce_sorted=False
        )
        
        H_o,H_t=self.rnn(H_packed)
        H_o,T=nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq
        )   
        #print(H_o)
        
        logits = self.linear(H_o)
        #シーケンス長がバラバラの場合:難しいので今は実装なし。
        """mask=[]
        for i in t:
        mask.append(torch.cat([torch.ones(i,10),torch.zeros(max_seq-i,10)]))
        mask=torch.stack(mask)
        H=mask*H"""
        
        return logits




class TimeGAN(nn.Module):
    def __init__(self,args):
        super(TimeGAN, self).__init__()
        self.device=args.device
        self.dim=args.dim
        self.hidden_dim=args.hidden_dim
        self.z_dim=args.z_dim
        self.max_seq=args.max_seq
        
        self.embedder=Embedding(args.dim,args.hidden_dim,args.num_layers,args.padding_value,args.max_seq)
        self.supervisor=Supervisor(args.hidden_dim,args.num_layers,args.padding_value,args.max_seq)
        self.recovery=Recovery(args.hidden_dim,args.dim,args.num_layers,args.padding_value,args.max_seq)
        self.generator=Generator(args.z_dim,args.hidden_dim,args.num_layers,args.padding_value,args.max_seq)
        self.discriminator=Discriminator(args.hidden_dim,args.num_layers,args.padding_value,args.max_seq)
    
    def recovery_loss(self,X,T):
        """
        recoveryloss+superviseloss

        out:
            E_loss: Recovery_Loss+Supervise_Loss
            E_loss0: sqrt(Recovery_+Loss)
            E_loss_T0: Recovery Loss

        """
        #embedding+recovery
        H=self.embedder(X,T)
        X_tilde=self.recovery(H,T)

        #embedding+supervise
        H_supervise=self.supervisor(H,T)
        
        E_loss_T0=nn.functional.mse_loss(X,X_tilde)
        G_loss_S=nn.functional.mse_loss(H_supervise[:,:-1,:],H[:,1:,:])

        E_loss0=10*torch.sqrt(E_loss_T0)
        E_loss=E_loss0+0.1*G_loss_S
        return E_loss,E_loss0,E_loss_T0
    
    def supervise_loss(self,X,T):
        """
        リアルデータを用いたsupervise_lossを計算
        """

        H=self.embedder(X,T)
        H_supervise=self.supervisor(H,T)

        G_loss_S=nn.functional.mse_loss(H_supervise[:,:-1,:],H[:,1:,:])
        return G_loss_S

    def discriminator_loss(self,X,T,Z,gamma=1):
        """
        ①リアルデータをembeddingしたデータを正例

        ② ①のデータをsuperviserで予測したものを負例

        ③ generatorでの生成データを負例

        gamma によって重みづけを行う。
        """
        H=self.embedder(X,T).detach()
        H_supervise=self.supervisor(H,T).detach()
        E=self.generator(Z,T).detach()

        real=self.discriminator(H,T).squeeze(-1)
        fake_h=self.discriminator(H_supervise,T).squeeze(-1)
        fake_e=self.discriminator(E,T).squeeze(-1)

        D_real_loss=nn.functional.binary_cross_entropy_with_logits(real,torch.ones_like(real))
        D_fake_h_loss=nn.functional.binary_cross_entropy_with_logits(fake_h,torch.zeros_like(fake_h))
        D_fake_e_loss=nn.functional.binary_cross_entropy_with_logits(fake_e,torch.zeros_like(fake_e))

        D_loss=D_real_loss+D_fake_h_loss+gamma*D_fake_e_loss

        return D_loss
        
    def generator_loss(self,X,T,Z,gamma=1):
        
        H=self.embedder(X,T)
        H_supervise=self.supervisor(H,T)
        X_tilde=self.recovery(H,T)

        E=self.generator(Z,T)
        E_supervise=self.supervisor(E,T)

        E_tilde_=self.recovery(E_supervise,T)

        fake=self.discriminator(E_supervise,T).squeeze(-1)
        fake_e=self.discriminator(E,T).squeeze(-1)

        G_A_loss=nn.functional.binary_cross_entropy_with_logits(fake,torch.ones_like(fake))
        G_A_e_loss=nn.functional.binary_cross_entropy_with_logits(fake_e,torch.ones_like(fake_e))

        G_loss_S=nn.functional.mse_loss(H_supervise[:,:-1,:],H[:,1:,:])

        G_loss_m1=torch.mean(torch.abs(torch.sqrt(E_tilde_.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_m2=torch.mean(torch.abs((E_tilde_.mean(dim=0)) - (X.mean(dim=0))))
        
        G_loss_m=G_loss_m1+G_loss_m2

        G_loss=G_A_loss+gamma*G_A_e_loss+100*torch.sqrt(G_loss_S)+100*G_loss_m

        return G_loss
    
    def generate(self,Z,T):

        H=self.generater(Z,T)
        H_=self.supervisor(H,T)

        X_g=self.recovery(H_,T)
        return X_g

    def forward(self,mode="generate",X=None,Z=None,T=None,gamma=1):
        """
        X must be tensor 
        """
        if T is None:
            raise ValueError("T must be given")

        if mode=="AutoEncoder":
            if X is None:
                raise ValueError(f"if mode: {mode} , X must be given")
            return self.recovery_loss(X,T)
        elif mode=="Supervise":
            if X is None:
                raise ValueError(f"if mode: {mode} , X must be given")
            return self.supervise_loss(X,T)
        elif mode=="Discriminator":
            if X is None:
                raise ValueError(f"if mode: {mode} , X must be given")
            if Z is None:
                raise ValueError(f"if mode: {mode} , Z must be given")
            return self.discriminator_loss(X,T,Z,gamma)
        elif mode=="Generator":
            if X is None:
                raise ValueError(f"if mode: {mode} , X must be given")
            if Z is None:
                raise ValueError(f"if mode: {mode} , Z must be given")
            return self.generator_loss(X,T,Z,gamma)
        elif mode=="generate":
            if Z is None:
                raise ValueError(f"if mode: {mode} , X must be given")
            X_g=self.generate(Z,T)
            return X_g.detach()
        
        else:
            raise ValueError(f"mode must be [ AutoEncoder , Supervise , Discriminator , Generator , generate]\n missing value:{mode}")

