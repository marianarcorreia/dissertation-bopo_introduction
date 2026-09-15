#import necessary libraries and modules for the environment. 
import copy #used for creating deep copies of the state to avoid unintended modifications   
import importlib
gym = importlib.import_module("gymnasium") #ambiente de rl
import numpy as np #mathematical operations and array manipulations
import random #to randomize action selection in the sample method
from torch_geometric.data import HeteroData #heterogeneous graph data structure from the PyTorch Geometric library, used to represent the state of the environment as a graph with different types of nodes and edges.
import torch #tensor and operation GPU/CPU
import os #reading environment variables for debugging purposes
import torch_geometric.transforms as T #graph transformations, not used in the current code but can be useful for data preprocessing or augmentation in graph-based environments.
from typing import Any
# Set environment variable FJSP_DEBUG to enable verbose prints.
# Levels: 1=env lifecycle  2=+step detail  3=+graph tensors
_DBG = int(os.environ.get("FJSP_DEBUG", "0")) #
#define the debug print function that checks the global debug level before printing. It takes a level and any number of arguments, and only prints if the global debug level is greater than or equal to the specified level. The printed message is prefixed with "[ENV]" to indicate that it comes from the environment code.
def _dbg(level, *args, **kwargs): #1 =env lifecycle  2=+step detail  3=+graph tensors
        if _DBG >= level:
                print("[ENV]", *args, **kwargs) 
#define the fjssp environment class that inherits from gym.Env. This class implements the logic for the Flexible Job Shop Scheduling Problem, including the state representation, action space, reward calculation, and episode management. The environment uses a heterogeneous graph to represent the state of the scheduling problem, with nodes for jobs, operations, and machines, and edges to represent their relationships and constraints.
class FJSSPEnv(gym.Env):
    def get_prev_op(self, o_id): #given an operation id, this method returns the previous operation in the same job, or None if it is the first operation. It loops through the list of jobs to find which job contains the given operation id, and then checks its position in that job to determine the previous operation.
        for j in self.jobs: #loop through jobs
            if o_id in j: # find the job that contains the operation
                if j.index(o_id) == 0: #get the position of o_id in that job
                    return None
                else: #return the previous operation in the job
                    return j[j.index(o_id) -1]
    #construtor for the FJSSPEnv class. It takes a list of problem instances, a mask option to determine how invalid actions are masked, and sel_k to determine how many candidate actions are selected by the mask. It initializes the base gym environment, stores the instances and parameters, and prints a debug message indicating that the environment has been created with the specified settings.
    def __init__(self, instances, mask_option = 3, sel_k = 5): #constructor for FJSSPEnv #aqui tenho 3
        super(FJSSPEnv, self).__init__() #initialize the gym.Env base class
        self.instances = instances #list of problem instances to solve
        self.current_instance = 0 #which instance is currently being solved
        self.mask_option = mask_option #how invalid actions are masked
        self.sel_k = sel_k #how many candidates are selected by the mask
        _dbg(1, f"FJSSPEnv created | {len(instances)} instance(s) | mask_option={mask_option} | sel_k={sel_k}") #debug message if debug level is 1, indicate the number of instance, masking mode, n of candidate action kept by mask

    def generate_instance(self, instance): #extrai os jobs and operatiosn
        jobs, operations = instance["jobs"], instance["operations"]
        _dbg(1, f"  generate_instance | jobs={len(jobs)} | operations={len(operations)} | machines={len(operations[0])}")
        #_dbg(level, +args,**kargs) is a debug print function that only prints if the global debug level is >= the specified level. Here it prints the number of jobs, operations and machines in the instance being generated.
        self.jobs = jobs #list of jobs, where each job is a list of operation ids
        self.num_jobs = len(jobs) 
        self.operations = operations
        self.num_operations = len(operations)
        self.num_machines = len(instance["operations"][0]) 
        
        #here defines the number of features for each type of node in the graph.
        self.num_features_job = 4
        self.num_features_oper = 2
        self.num_features_mach = 3
       
        self.max_change = int(len(operations))#define the maximum number of steps in an episode, which is set to the total number of operations. This means that in the worst case, the agent will have to schedule all operations one by one to complete all jobs.
        self.data = HeteroData()#create an empty heterogeneous graph data structure to represent the state of the environment.
        #initialize the node features for each type of node in the graph.
        #it will be updated in the reset and calculate_next_state methods to reflect the current state of the environment, such as which operations are pending, which machines are occupied, and how much time has passed.
        self.data["job"].x = torch.zeros((self.num_jobs, self.num_features_job), dtype= torch.float)
        self.data["operation"].x = torch.zeros((len(self.operations), self.num_features_oper), dtype= torch.float)
        self.data["machine"].x = torch.zeros((self.num_machines, self.num_features_mach), dtype= torch.float)
        
        #arestas de precedencia entre operações do mesmo job, com self-loops
        aux_list = []
        for i in range(self.num_jobs):
            for j in self.jobs[i]:
                aux_list.append([j, i]) # operation -> job
        
        self.data['operation', 'belongs', 'job'].edge_index = torch.LongTensor(aux_list).T
        
        for j in self.jobs:
            for i in range(len(j)-1): #tem -1  pq não faz a última
                aux_list.append([j[i], j[i]]) #self-loop
                aux_list.append([j[i+1], j[i]]) #precendencia
            aux_list.append([j[-1], j[-1]]) #self loop da  ultima operação

        self.data['operation', 'prec', 'operation'].edge_index = torch.LongTensor(aux_list).T

        #deadcode
        aux_list = []
        aux_list2 = []
        for j in range(self.num_jobs):
            for i in range(self.num_machines):
                aux_list.append([i, j])
                aux_list2.append([j,i])

        #cria arestas entre todas as máquinas (+ self-loops)
        aux_list = []
        for i in range(self.num_machines):
            for j in range(self.num_machines):
                aux_list.append([i, j])
        self.data['machine', 'listens', 'machine'].edge_index = torch.LongTensor(aux_list).T

        aux_list = []
        for i in range(self.num_jobs):
            for j in range(self.num_jobs):
                aux_list.append([i, j])
        self.data['job', 'listens', 'job'].edge_index = torch.LongTensor(aux_list).T

        #tempo de processamento pendente para cada operação (soma dos tempos das operações seguintes no mesmo job) → usado para calcular features e rewards
        self.all_pendings  = [] 
        for os in jobs:
            aux = []
            #aux_o = np.array(operations[j])
            for o_id in reversed(os): #percurre a operação do fim para o inicio
                aux_o = np.array(operations[o_id])
                try:
                    aux.append(np.mean(aux_o[np.where(aux_o!=0)]) + aux[-1])  #está a acumular o tempo
                except:
                    aux.append(np.mean(aux_o[np.where(aux_o!=0)]))
            self.all_pendings = self.all_pendings + list(reversed(aux))
        
        #cria arestas bidirecionais com 5 features (t, racio de t, racio em rel t pend acumulado, 2x 0)
        aux_list = []
        aux_list_2 = []
        aux_list_features = []
        for i in range(len(self.operations)):
            o = self.operations[i]
            for j in range(len(o)):
                t = o[j]
                if t!=0:    #apenas para máquinas compativeis
                    aux_list.append([i, j])
                    aux_list_2.append([j, i])
                    aux_list_features.append([t, t/np.sum(o), t/self.all_pendings[i], 0, 0])
        #guarda as arestas entre operações e máquinas, com suas respectivas features, no grafo de dados. As arestas do tipo 'operation', 'exec', 'machine' representam a relação de que uma operação pode ser executada em uma máquina específica, enquanto as arestas do tipo 'machine', 'exec', 'operation' representam a relação inversa. As features associadas a essas arestas incluem o tempo de processamento, o rácio do tempo em relação ao total, o rácio do tempo em relação ao tempo pendente acumulado, e dois valores iniciais de 0 que podem ser atualizados posteriormente para refletir o estado atual da execução.
        self.data['operation', 'exec', 'machine'].edge_index = torch.LongTensor(aux_list).T
        self.data['operation', 'exec', 'machine'].edge_attr = torch.Tensor(aux_list_features)

        self.data['machine', 'exec', 'operation'].edge_index = torch.LongTensor(aux_list_2).T
        self.data['machine', 'exec', 'operation'].edge_attr = torch.Tensor(aux_list_features)

        #DEBUG: print the number of edges and node types in the heterogeneous graph after initialization. This helps to verify that the graph has been constructed correctly with the expected number of nodes and edges based on the input instance.
        _dbg(3, f"    HeteroData edges: op->mach={len(aux_list)} | node types: op={self.num_operations}, mach={self.num_machines}")
        
        #preenche a feature de tempo pendente acumulado para cada operação, que é a soma dos tempos das operações seguintes no mesmo job. Isso é feito para cada operação em cada job, percorrendo as operações do fim para o início e acumulando o tempo. Essa informação é importante para calcular as features e os rewards durante a execução do ambiente.
        for i in range(self.num_jobs):
            o_index = 0
            for j in self.jobs[i]:
                self.data["operation"].x[j, 1] = self.all_pendings[j]
                aux = np.array(operations[j])
                o_index+=1


    #reset
    def reset(self, sel_index = None):
        idx = sel_index if sel_index is not None else self.current_instance
        _dbg(1, f"  reset() | instance_index={idx}")
        #gera a instancia e avança com o índice da instância atual para a próxima. Se sel_index for fornecido, ele gera a instância correspondente a esse índice em vez de usar o índice atual. Isso permite que o ambiente seja reiniciado com uma instância específica, o que pode ser útil para testes ou para garantir a consistência durante o treinamento.
        if sel_index is None:
            self.generate_instance(self.instances[self.current_instance])
            self.current_instance = (self.current_instance + 1)%len(self.instances)
        else:
            self.generate_instance(self.instances[sel_index])
        #reinica para estado inicial
        self.num_steps = 0
        self.change_machine = 0
        self.state: Any = copy.deepcopy(self.data)

        self.job_start_machines = torch.empty((self.num_jobs,self.num_machines)) #o instante que apartir dai o job j começa na máquina m
        self.current_job_proc = torch.zeros((self.num_jobs,self.num_machines)) #o tempo de processamento

        self.current_operations = [0]*self.num_jobs #os job que estiveram no indice 0, serão os atuais agora
        for j_id in range(len(self.jobs)):
            self.current_operations[j_id] = self.jobs[j_id][0]
        
        #para atualizar
        self.operations_ends = [0]*self.num_jobs #o instante que cada job termina a última operação feita, usado para calcular features e rewards
        self.machines_occupations = [0]*self.num_machines #tempo total que cada máquina esteve ocupada, usado para calcular features e rewards
        
        #arestas inicias máquina -job
        aux_list = []
        aux_list_features = []
        for j_id in range(len(self.jobs)):
            oper = self.operations[self.jobs[j_id][0]] 
            for m in range(len(oper)):
                t = oper[m] #tempo de processamento da operação atual do job j_id na máquina m
                if t!=0: #apenas para máquinas compativeis
                    aux_list.append([m, j_id])  # máquina -> operação atual do job
                    aux_list_features.append([t, t/np.sum(oper), t/np.sum(oper), 0, t])
                    self.job_start_machines[j_id,m] = 0
                    self.current_job_proc[j_id,m] = int(t)
                else:
                    self.job_start_machines[j_id,m] = 10000 #para que nunca seja utilizadas
        #guarda as arestas iniciais entre máquinas e jobs, com suas respectivas features, no grafo de dados. Essas arestas representam as possíveis atribuições iniciais de operações aos jobs, e as features associadas incluem o tempo de processamento, o rácio do tempo em relação ao total, o rácio do tempo em relação ao tempo pendente acumulado, um valor inicial de 0 que pode ser atualizado posteriormente, e o tempo de processamento da operação atual do job na máquina. A matriz job_start_machines é preenchida para indicar o instante em que cada job pode começar a ser processado em cada máquina, com um valor alto (10000) para máquinas incompatíveis para garantir que elas nunca sejam selecionadas.
        self.state['machine', 'exec', 'job'].edge_index = torch.LongTensor(aux_list).T
        self.state['machine', 'exec', 'job'].edge_attr = torch.Tensor(aux_list_features)

        #atualiza o estado inicial calculando as features e a máscara de ações válidas. A função calculate_next_state() é chamada para atualizar as features dos nós e arestas com base no estado inicial, como o tempo pendente acumulado, o tempo de ocupação das máquinas, e outros atributos relevantes para a tomada de decisão. Em seguida, a função calculate_mask() é chamada para determinar quais ações são válidas com base no estado atual, aplicando a lógica de mascaramento definida pela opção de máscara selecionada. Isso garante que o agente só possa escolher ações que sejam viáveis no contexto do problema de escalonamento.
        self.calculate_next_state()
        self.calculate_mask()

        _dbg(1, f"  reset() done | {self.num_jobs} jobs | {self.num_operations} ops | {self.num_machines} machines | pending_edges={self.state['machine','exec','job'].edge_index.shape[1]}")
        return self.state
    #definir o critério de escalonamento
    def calculate_mask(self):
        _dbg(2, f"    calculate_mask() | mask_option={self.mask_option} | sel_k={self.sel_k}")
        if self.mask_option == 0: #este diz para escolher as que começam + cedo, se fosse mask_option != 0 seria as que terminam mais cedo
            mask_matrix = self.job_start_machines
        else:
            mask_matrix = self.job_start_machines + self.current_job_proc
        
        smallest = torch.unique(torch.topk(torch.flatten(mask_matrix), k = self.sel_k, largest = False, dim = 0).values) #encontra os sel_k menores, mais promissores
        min_jobs = torch.tensor([], dtype=torch.long)
        min_machines = torch.tensor([], dtype=torch.long)

        for s in smallest: #encontra que jobs e máquinas correspondem a esses menores valores, ou seja, quais são os candidatos mais promissores para serem escalonados a seguir com base no critério selecionado. Ele percorre os valores únicos dos menores tempos encontrados e utiliza a função nonzero para obter as posições (índices) no mask_matrix onde esses valores ocorrem. Os índices das máquinas e dos jobs correspondentes a esses valores são então concatenados em min_machines e min_jobs, respectivamente, para formar uma lista de candidatos válidos para a próxima ação.
            mj , mm = (mask_matrix == s).nonzero(as_tuple=True)
            min_machines = torch.concat([min_machines, mm])
            min_jobs = torch.concat([min_jobs, mj])

        #cada par sel, encontra se o índice da aresta correspondente no grafo de dados 
        pairs = torch.stack([min_machines, min_jobs]).T
        indexes = []
        for p in pairs:
            aux =self.state['machine', 'exec', 'job'].edge_index.T == p
            aux = np.logical_and(aux[:,0], aux[:,1])
            indexes = indexes + [i for i, val in enumerate(aux) if val==1] 

        #cria mascara final, true-açao bloqueada, false-açao válida.
        res = [True]*self.state['machine', 'exec', 'job'].edge_index.shape[1]
        for i in indexes:
            res[i] = False
        
        self.state['machine', 'exec', 'job'].mask = torch.BoolTensor(res)
        _dbg(2, f"    calculate_mask() done | total_edges={len(res)} | unmasked={res.count(False)} | masked={res.count(True)}")
    
    def calculate_next_state(self):
        #atualiza a ft 2 das maq. - tempo livre relativo ao mínimo
        self.state["machine"].x[:,2] = self.state["machine"].x[:,0] - torch.min(self.state["machine"].x[:,0])
        for j_id in range(len(self.jobs)):
            if int(self.state["job"].x[j_id,0])==0: #se o job ainda não tiver terminado
                o_id = self.current_operations[j_id] #operação atual do job j_id
                pj = self.state['operation', 'belongs', 'job'].edge_index[:,self.state['operation', 'belongs', 'job'].edge_index[1,:] == j_id]
                oper_id = pj[0,0]
                self.state["operation"].x[oper_id, 0] = 1
                #self.state["operation"].x[oper_id, 1] = self.all_pendings[o_id]
                self.state["job"].x[j_id, 1] = self.operations_ends[j_id]
                self.state["job"].x[j_id, 2] = pj.shape[1]
                self.state["job"].x[j_id, 3] = self.all_pendings[o_id]
        
        #atualiza as features das máquinas (ft4)
        for m in range(len(self.state["machine"].x)):
            mask = self.state["operation", "exec", "machine"].edge_index[1,:] == m
            if mask.any().item():
                self.state["operation", "exec", "machine"].edge_attr[mask,4] = self.state["operation", "exec", "machine"].edge_attr[mask,0]/self.state["operation", "exec", "machine"].edge_attr[mask,0].max()
                mask = self.state["machine", "exec", "operation"].edge_index[0,:] == m
                self.state["machine", "exec", "operation"].edge_attr[mask,4] = self.state["machine", "exec", "operation"].edge_attr[mask,0]/self.state["machine", "exec", "operation"].edge_attr[mask,0].max()

            mask = self.state["machine", "exec", "operation"].edge_index[0,:] == m
            if mask.any().item():
                self.state["machine", "exec", "operation"].edge_attr[mask,4] = self.state["machine", "exec", "operation"].edge_attr[mask,0]/self.state["machine", "exec", "operation"].edge_attr[mask,0].max()
                #self.state["machine"].x[m, 7] = self.state["machine", "exec", "operation"].edge_attr[mask,0].shape[0]                   
    
    def step(self, action):
        #converte o indice da ação no par (m, j)
        self.num_steps+=1       
        
        action = self.state['machine', 'exec', 'job'].edge_index[:,action]
        
        sel_job = int(action[1])
        sel_mach = int(action[0])
        _dbg(2, f"  step #{self.num_steps} | sel_job={int(sel_job)} | sel_mach={int(sel_mach)}")
        #remoção da aresta máquina-job selecionada
        mask = self.state['machine', 'exec', 'job'].edge_index[1,:] != sel_job
        self.state['machine', 'exec', 'job'].edge_index = self.state['machine', 'exec', 'job'].edge_index[:,mask]
        self.state['machine', 'exec', 'job'].edge_attr = self.state['machine', 'exec', 'job'].edge_attr[mask]
        
        #guarda o makespan anterior para calcular o reward depois
        prev_ms = float(torch.max(self.state["machine"].x[:,0]))
        o_id = self.current_operations[sel_job] #operação atual do job selecionado

        #calcula os tempos
        start_time = max(self.state["machine"].x[sel_mach,0], self.operations_ends[sel_job])
        proc_time  = self.operations[o_id][sel_mach]
        #o tempo de conclusão
        final_time =  start_time + proc_time
        #atualiza o tempo livre da máquina selecionada, que é o tempo em que a máquina estará disponível para a próxima operação. Isso é calculado como o tempo de conclusão da operação atual (final_time), e é armazenado na feature 0 da máquina selecionada no estado do ambiente. Essa atualização é crucial para refletir o impacto da ação tomada no estado do ambiente, permitindo que o agente tome decisões informadas nas próximas etapas com base na disponibilidade das máquinas.
        self.state["machine"].x[sel_mach, 0] = final_time
        #atualiza os tempos de início possíveis para todos os outros jobs na máquina selecionada, garantindo que eles não possam ser agendados para começar antes que a máquina esteja disponível. Isso é feito comparando o tempo de início atual para cada job na máquina selecionada (armazenado em self.job_start_machines) com o tempo de conclusão da operação atual (final_time), e atualizando o tempo de início para ser no mínimo igual a final_time. Essa lógica garante que as restrições de disponibilidade da máquina sejam respeitadas para todos os jobs que possam ser agendados na mesma máquina.
        self.job_start_machines[self.job_start_machines[:,sel_mach] <  final_time, sel_mach] =  final_time
        #atualiza a taxa de utilização da máquina 
        self.machines_occupations[sel_mach] += proc_time
        self.state["machine"].x[sel_mach, 1] = self.machines_occupations[sel_mach]/final_time
        #registo do tempo de conclusão do job
        self.operations_ends[sel_job] = final_time
        self.job_start_machines[sel_job,:] = 10000
        self.current_job_proc[sel_job, :] = 0
        #se esta for a última operação do job selecionado, marca o job como concluído e remove as arestas de escuta entre esse job e os outros jobs. Isso é feito verificando se a operação atual do job selecionado é a última operação na lista de operações para esse job. Se for o caso, a feature 0 do job selecionado é definida como 1 para indicar que o job foi concluído, e as arestas do tipo 'job', 'listens', 'job' que conectam o job selecionado a outros jobs são removidas do grafo de dados. Essa lógica garante que uma vez que um job seja concluído, ele não possa mais ser considerado para agendamento ou influenciar outros jobs no ambiente.
        
        if (int(self.current_operations[sel_job])) == self.jobs[sel_job][-1]:
            self.state["job"].x[sel_job,:]=0
            self.state["job"].x[sel_job,0]=1

            self.state['job', 'listens', 'job'].edge_index = self.state['job', 'listens', 'job'].edge_index[:, self.state['job', 'listens', 'job'].edge_index[0,:] != sel_job]
 
        #avança para a próxima
        else:
            self.current_operations[sel_job]+=1
            aux_list = []
            aux_list_features = []
            oper= np.array(self.operations[self.current_operations[sel_job]])
            total_gap = 0
            for m in range(len(oper)):
                t = oper[m]
                #atualiza as maq. compativeis, atualiza job_start_machines e cria novas arestas
                if t!=0:
                    calcu = t + max(self.operations_ends[sel_job] - self.state["machine"].x[m, 0],0)
                    total_gap += calcu
                    aux_list.append([m, sel_job])
                    aux_list_features.append([calcu, t/np.sum(oper) , t + max(self.operations_ends[sel_job], self.state["machine"].x[m, 0]) ])
                    self.job_start_machines[sel_job,m] = max(self.operations_ends[sel_job], self.state["machine"].x[m, 0])
                    self.current_job_proc[sel_job,m] = t
            #adiciona as novas arestas e features no grafo existente
            for l in aux_list_features:
                l.append(l[0]/total_gap) #ft4: racio do gap em rel ao total
                l.append(0) #ft5: valor inical

            self.state['machine', 'exec', 'job'].edge_index = torch.concat([
                self.state['machine', 'exec', 'job'].edge_index,
                torch.LongTensor(aux_list).T
            ], dim=1)
            self.state['machine', 'exec', 'job'].edge_attr = torch.concat([
                self.state['machine', 'exec', 'job'].edge_attr,
                torch.Tensor(aux_list_features)
            ], dim=0)
        #reward cal. Se a ms não cresceu, recompensa positiva, se cresceu, recompensa negativa.
        reward = prev_ms - float(torch.max(self.state["machine"].x[:,0])) 
        
        #se todos os jobs concluidos, o episodio termina. Makespan fica me self.mk
        if torch.all(self.state["job"].x[:,0]==1):
            self.mk = round(float(torch.max(self.state["machine"].x[:,0])),2)
            _dbg(1, f"  episode DONE after {self.num_steps} steps | makespan={self.mk} | reward_last={reward:.4f}")
            return self.state, reward , True, {"current_machine": sel_mach}
            
        #remove a operação escalonada de todas as est.
        oper_id = self.state['operation', 'belongs', 'job'].edge_index[:,self.state['operation', 'belongs', 'job'].edge_index[1,:] == sel_job][0,0]
        self.state['operation', 'belongs', 'job'].edge_index = self.state['operation','belongs', 'job'].edge_index[:,self.state['operation', 'belongs', 'job'].edge_index[0,:] != oper_id]
        self.state['operation', 'prec', 'operation'].edge_index = self.state['operation', 'prec', 'operation'].edge_index[:,self.state['operation', 'prec', 'operation'].edge_index[1,:] != oper_id]
        
        mask = self.state['operation', 'exec', 'machine'].edge_index[0,:] != oper_id
        self.state['operation', 'exec', 'machine'].edge_index = self.state['operation', 'exec', 'machine'].edge_index[:,mask]
        self.state['operation', 'exec', 'machine'].edge_attr = self.state['operation', 'exec', 'machine'].edge_attr[mask]

        mask = self.state['machine', 'exec', 'operation'].edge_index[1,:] != oper_id
        self.state['machine', 'exec', 'operation'].edge_index = self.state['machine', 'exec', 'operation'].edge_index[:,mask]
        self.state['machine', 'exec', 'operation'].edge_attr = self.state['machine', 'exec', 'operation'].edge_attr[mask]
        self.state = T.RemoveIsolatedNodes()(self.state)
        #atualiza features e recalcula a mascara
        self.calculate_next_state()
        self.calculate_mask()

        done = False
        total_reward = reward
        if len(self.state['machine', 'exec', 'job'].mask) - sum(self.state['machine', 'exec', 'job'].mask)==1:
            self.state, reward , done, _ = self.step(self.sample())
            total_reward +=reward
        _dbg(2, f"    active edges after step={int(len(self.state['machine', 'exec', 'job'].mask) - sum(self.state['machine', 'exec', 'job'].mask))} | reward={reward:.4f} | curr_makespan={float(torch.max(self.state['machine'].x[:,0])):.2f}")
        #se apenas faltar uma ação válida, está será executada sem precisar de chamar a politica
        return self.state, total_reward , done, {"current_machine": sel_mach}
    
    #escolhe aleatoriamente uma ação
    def sample(self):
        return random.choice([i for i in range(len(self.state['machine', 'exec', 'job'].mask)) if not self.state['machine', 'exec', 'job'].mask[i]])
    
    #normaliza as features dos nós e arestas para o intervalo [-1, 1]. Isso é feito para cada tipo de nó (job, operation, machine) e para as arestas entre operações e máquinas. A normalização é realizada usando a fórmula (2*(x - min)/(max - min + 1e-7) - 1), onde x é o valor da feature, min é o valor mínimo da feature no conjunto de dados, max é o valor máximo da feature no conjunto de dados, e 1e-7 é um pequeno valor adicionado para evitar divisão por zero. Essa normalização ajuda a estabilizar o treinamento do agente de aprendizado por reforço, garantindo que as features estejam em uma escala consistente.
    def normalize_state(self, state):
        state = copy.deepcopy(state)
        for i in range(state["job"].x.shape[1]):
            state["job"].x[:,i] = (2*(state["job"].x[:,i] - state["job"].x[:,i].min())/(state["job"].x[:,i].max() - state["job"].x[:,i].min() + 1e-7 )-1).float()
        
        for i in range(state["operation"].x.shape[1]):
            state["operation"].x[:,i] = (2*(state["operation"].x[:,i] - state["operation"].x[:,i].min())/(state["operation"].x[:,i].max() - state["operation"].x[:,i].min() + 1e-7 )-1).float()
        
        for i in range(state["machine"].x.shape[1]):
            state["machine"].x[:,i] = (2*(state["machine"].x[:,i] - state["machine"].x[:,i].min())/(state["machine"].x[:,i].max() - state["machine"].x[:,i].min() + 1e-7 )-1).float()

        state[('operation', 'exec', 'machine')].edge_attr = (2*(state[('operation', 'exec', 'machine')].edge_attr -  state[('operation', 'exec', 'machine')].edge_attr.min())/(state[('operation', 'exec', 'machine')].edge_attr.max() - state[('operation', 'exec', 'machine')].edge_attr.min() + 1e-7 )-1).float()
        state[('machine', 'exec', 'operation')].edge_attr = (2*(state[('machine', 'exec', 'operation')].edge_attr -  state[('machine', 'exec', 'operation')].edge_attr.min())/(state[('machine', 'exec', 'operation')].edge_attr.max() - state['machine', 'exec', 'operation'].edge_attr.min() + 1e-7 )-1).float()
        state[('machine', 'exec', 'job')].edge_attr = (2*(state[('machine', 'exec', 'job')].edge_attr - state[('machine', 'exec', 'job')].edge_attr.min())/(state[('machine', 'exec', 'job')].edge_attr.max() - state[('machine', 'exec', 'job')].edge_attr.min() + 1e-7 )-1).float()
        return state
    
