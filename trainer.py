from sklearn.model_selection import train_test_split
import torch
from Dataset.dataset import Time_Dataset
class Trainer:
    def __init__(self,args,model,data,time,max_seq):
        self.args=args
        self.model=model.to(args.device)
        self.data=data
        self.time=time
        self.max_seq=max_seq
        self.split_data(args.train_rate)
        self.train_dataset=Time_Dataset(self.train_data,self.train_time)
        self.train_loader=torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True    
        )
        self.test_dataset=Time_Dataset(self.test_data,self.test_time)
        self.test_loader=torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=args.test_batch_size,
            shuffle=True    
        )
        self.lr=args.lr
        self.e_opt=torch.optim.Adam(self.model.embedder.parameters(),lr=self.lr)
        self.s_opt=torch.optim.Adam(self.model.supervisor.parameters(),lr=self.lr)
        self.r_opt=torch.optim.Adam(self.model.recovery.parameters(),lr=self.lr)
        self.g_opt=torch.optim.Adam(self.model.generator.parameters(),lr=self.lr)
        self.d_opt=torch.optim.Adam(self.model.discriminator.parameters(),lr=self.lr)
        

    def split_data(self,train_rate=0.7):
        self.train_data,self.test_data,self.train_time,self.test_time=train_test_split(self.data,self.time,train_size=1-train_rate)

    def embedding_train(self):
        #embedderとrecoveryの学習を行う。ただこの段階では再構成誤差のみ
        for epoch in range(self.args.e_epochs):
            for d,ti in self.train_loader:
                d=d.to(self.args.device)
                ti=ti.to(self.args.device)
                self.model.zero_grad()
                E_loss,E_loss0,E_loss_T0=self.model(mode="AutoEncoder",X=d,T=ti)
                E_loss0.backward()

                self.e_opt.step()
                self.r_opt.step()
            if epoch%20==0:
                print(f"Embedding Train Step [{epoch}] loss : {E_loss_T0.item()}")

    
    def supervise_train(self):
        #中間層のH_(t-1)からH_(t)を予測するモジュールの学習。再構成誤差
        for epoch in range(self.args.s_epochs):
            for d, ti in self.train_loader:
                d=d.to(self.args.device)
                ti=ti.to(self.args.device)
                self.model.zero_grad()
                G_loss_S=self.model(mode="Supervise",X=d,T=ti)
                G_loss_S.backward()

                self.s_opt.step()
            if epoch%20==0:
                print(f"Surpervise Train Step [{epoch}] loss : {G_loss_S.item()}") 
    
    def GAN_train(self):
        for epoch in range(self.args.g_epochs):
            for d, ti in self.train_loader:
                d=d.to(self.args.device)
                ti=ti.to(self.args.device)
                for _ in range(2):
                    #まず、superviserとgeneratorの学習。敵対誤差、再構成誤差を含む
                    self.model.zero_grad()
                    Z=torch.randn(d.shape[0],d.shape[1],self.args.z_dim)
                    G_loss=self.model(mode="Generator",X=d,T=ti,Z=Z)
                    G_loss.backward()

                    self.g_opt.step()
                    self.s_opt.step()

                    #再構成誤差と予測モジュールとの再構成誤差の最適化。ただし、sは学習しない。
                    self.model.zero_grad()
                    E_loss,E_loss0,E_loss_T0=self.model(mode="AutoEncoder",X=d,T=ti)
                    E_loss.backward()

                    self.e_opt.step()
                    self.r_opt.step()

                self.model.zero_grad()
                Z=torch.randn(d.shape[0],d.shape[1],self.args.z_dim)
                
                D_loss=self.model(mode="Discriminator",X=d,T=ti,Z=Z)
                if D_loss > self.args.dis_check:
                    D_loss.backward()
                    self.d_opt.step()
            if epoch%20==0:
                print(f"Generator Train Step [{epoch}] loss : {G_loss.item()}")
                print(f"Embedder Train Step [{epoch}] loss : {E_loss.item()}")
                print(f"Discriminator Train Step [{epoch}] loss : {D_loss.item()}")
                
    def train(self):
        print(self.args)
        print("Embedding Training Start")
        self.embedding_train()
        print("Superviser Training Start ")
        self.supervise_train()
        print("Last Training Start")
        self.GAN_train()
        print("finish training")
        torch.save(self.model.state_dict(),"./timegan.pt")
        print("Saved!")





                

