%matplotlib
import pandas as pd
import matplotlib.pyplot as plt 
import hddm

print hddm.__version__

## load data
data = hddm.load_csv('./ANT_Final.csv')

##Use seconds not millisecs
data['rt'] = data['rt'] / 1000

#MODELS
#Model 1
#Model comparison one, only v by stimulus
m = hddm.HDDMStimCoding(data, group_only_nodes=['v','a','t'], stim_col='stimulus', split_param='v', depends_on={'v': ['group','stim_2'],'a':'group','t': 'group'},  p_outlier=0.05) 
m.find_starting_values() 
m.sample(120000, burn=20000, thin=10, dbname='traces.db', db='pickle') 
m.save('Stimcode_model_1')

#Model 2
#Model comparison two, only a by stimulus
m = hddm.HDDMStimCoding(data, group_only_nodes=['v','a','t'], stim_col='stimulus', split_param='v', depends_on={'v': 'group','a':['group', 'stim_2'],'t': 'group'},  p_outlier=0.05) 
m.find_starting_values() 
m.sample(120000, burn=20000, thin=10, dbname='traces.db', db='pickle') 
m.save('Stimcode_model_2')

#Model 3
#Full model with v and a by stim
m = hddm.HDDMStimCoding(data, group_only_nodes=['v','a','t'], stim_col='stimulus', split_param='v', depends_on={'v': ['group','stim_2'],'a':['group', 'stim_2'],'t': 'group'},  p_outlier=0.05) 
m.find_starting_values() 
m.sample(120000, burn=20000, thin=10, dbname='traces.db', db='pickle') 
m.save('Stimcode_model_3')

#Inspect traces
m.plot_posteriors('a')
m.plot_posteriors('t')
m.plot_posteriors('v')

#Plot posterior predictives
m.plot_posterior_predictive(figsize=(10, 8))

#CONVERGENCE CHECKS
#Gelman model 1
models = []
for i in range(5):
    m = hddm.HDDMStimCoding(data, group_only_nodes=['v','a','t'], stim_col='stimulus', split_param='v', depends_on={'v': ['group','stim_2'],'a':'group','t': 'group'},  p_outlier=0.05)
    m.find_starting_values()
    m.sample(120000, burn=20000, thin=10)
    models.append(m)

hddm.analyze.gelman_rubin(models)

#Gelman model 2
models = []
for i in range(5):
    m = hddm.HDDMStimCoding(data, group_only_nodes=['v','a','t'], stim_col='stimulus', split_param='v', depends_on={'v': 'group','a':['group', 'stim_2'],'t': 'group'},  p_outlier=0.05)
    m.find_starting_values()
    m.sample(120000, burn=20000, thin=10)
    models.append(m)

hddm.analyze.gelman_rubin(models)

#Gelman model 3
models = []
for i in range(5):
   m = hddm.HDDMStimCoding(data, group_only_nodes=['v','a','t'], stim_col='stimulus', split_param='v', depends_on={'v': ['group','stim_2'],'a':['group', 'stim_2'],'t': 'group'},  p_outlier=0.05) 
    m.find_starting_values()
    m.sample(120000, burn=20000, thin=10)
    models.append(m)

hddm.analyze.gelman_rubin(models)


#Posterior predictive check
ppc_data = hddm.utils.post_pred_gen(m)
ppc_compare = hddm.utils.post_pred_stats(data, ppc_data)
ppc_stats = hddm.utils.post_pred_stats(data, ppc_data, call_compare=False)


########Hypothesis testing#################

#Between group drift rate comparisons
#Incongruent
Control_v_incongruent, nonVH_v_incongruent = m.nodes_db.node[['v(Control.incongruent)', 'v(nonVH.incongruent)']]
print "P_v(Control_v_incongruent > nonVH_v_incongruent) =", (Control_v_incongruent.trace()> nonVH_v_incongruent.trace()).mean()
Control_v_incongruent, VH_v_incongruent = m.nodes_db.node[['v(Control.incongruent)', 'v(VH.incongruent)']]
print "P_v(Control_v_incongruent > VH_v_incongruent) =", (Control_v_incongruent.trace()> VH_v_incongruent.trace()).mean()
nonVH_v_incongruent, VH_v_incongruent = m.nodes_db.node[['v(nonVH.incongruent)', 'v(VH.incongruent)']]
print "P_v(nonVH_v_incongruent > VH_v_incongruent) =", (nonVH_v_incongruent.trace()> VH_v_incongruent.trace()).mean()

#Congruent
Control_v_congruent, nonVH_v_congruent = m.nodes_db.node[['v(Control.congruent)', 'v(nonVH.congruent)']]
print "P_v(Control_v_congruent > nonVH_v_congruent) =", (Control_v_congruent.trace()> nonVH_v_congruent.trace()).mean()
Control_v_congruent, VH_v_congruent = m.nodes_db.node[['v(Control.congruent)', 'v(VH.congruent)']]
print "P_v(Control_v_congruent > VH_v_congruent) =", (Control_v_congruent.trace()> VH_v_congruent.trace()).mean()
nonVH_v_congruent, VH_v_congruent = m.nodes_db.node[['v(nonVH.congruent)', 'v(VH.congruent)']]
print "P_v(nonVH_v_congruent > VH_v_congruent) =", (nonVH_v_congruent.trace()> VH_v_congruent.trace()).mean()

#Neutral
Control_v_neutral, nonVH_v_neutral = m.nodes_db.node[['v(Control.neutral)', 'v(nonVH.neutral)']]
print "P_v(Control_v_neutral > nonVH_v_neutral) =", (Control_v_neutral.trace()> nonVH_v_neutral.trace()).mean()
Control_v_neutral, VH_v_neutral = m.nodes_db.node[['v(Control.neutral)', 'v(VH.neutral)']]
print "P_v(Control_v_neutral > VH_v_neutral) =", (Control_v_neutral.trace()> VH_v_neutral.trace()).mean()
nonVH_v_neutral, VH_v_neutral = m.nodes_db.node[['v(nonVH.neutral)', 'v(VH.neutral)']]
print "P_v(nonVH_v_neutral > VH_v_neutral) =", (nonVH_v_neutral.trace()> VH_v_neutral.trace()).mean()

#Between group decision boundary comparisons
#Incongruent
Control_a_incongruent, nonVH_a_incongruent = m.nodes_db.node[['a(Control.incongruent)', 'a(nonVH.incongruent)']]
print "P_a(Control_a_incongruent < nonVH_a_incongruent) =", (Control_a_incongruent.trace()< nonVH_a_incongruent.trace()).mean()
Control_a_incongruent, VH_a_incongruent = m.nodes_db.node[['a(Control.incongruent)', 'a(VH.incongruent)']]
print "P_a(Control_a_incongruent < VH_a_incongruent) =", (Control_a_incongruent.trace()< VH_a_incongruent.trace()).mean()
nonVH_a_incongruent, VH_a_incongruent = m.nodes_db.node[['a(nonVH.incongruent)', 'a(VH.incongruent)']]
print "P_a(nonVH_a_incongruent > VH_a_incongruent) =", (nonVH_a_incongruent.trace()> VH_a_incongruent.trace()).mean()

#Congruent
Control_a_congruent, nonVH_a_congruent = m.nodes_db.node[['a(Control.congruent)', 'a(nonVH.congruent)']]
print "P_a(Control_a_congruent < nonVH_a_congruent) =", (Control_a_congruent.trace()< nonVH_a_congruent.trace()).mean()
Control_a_congruent, VH_a_congruent = m.nodes_db.node[['a(Control.congruent)', 'a(VH.congruent)']]
print "P_a(Control_a_congruent < VH_a_congruent) =", (Control_a_congruent.trace()< VH_a_congruent.trace()).mean()
nonVH_a_congruent, VH_a_congruent = m.nodes_db.node[['a(nonVH.congruent)', 'a(VH.congruent)']]
print "P_a(nonVH_a_congruent > VH_a_congruent) =", (nonVH_a_congruent.trace()> VH_a_congruent.trace()).mean()

#Neutral
Control_a_neutral, nonVH_a_neutral = m.nodes_db.node[['a(Control.neutral)', 'a(nonVH.neutral)']]
print "P_a(Control_a_neutral < nonVH_a_neutral) =", (Control_a_neutral.trace()< nonVH_a_neutral.trace()).mean()
Control_a_neutral, VH_a_neutral = m.nodes_db.node[['a(Control.neutral)', 'a(VH.neutral)']]
print "P_a(Control_a_neutral < VH_a_neutral) =", (Control_a_neutral.trace()< VH_a_neutral.trace()).mean()
nonVH_a_neutral, VH_a_neutral = m.nodes_db.node[['a(nonVH.neutral)', 'a(VH.neutral)']]
print "P_a(nonVH_a_neutral > VH_a_neutral) =", (nonVH_a_neutral.trace()> VH_a_neutral.trace()).mean()

#Non decision time
Control_t, nonVH_t = m.nodes_db.node[['t(Control)', 't(nonVH)']]
print "P_t(Control_t > nonVH_t) =", (Control_t.trace()> nonVH_t.trace()).mean()
Control_t, VH_t = m.nodes_db.node[['t(Control)', 't(VH)']]
print "P_t(Control_t > VH_t) =", (Control_t.trace()> VH_t.trace()).mean()
nonVH_t, VH_t = m.nodes_db.node[['t(nonVH)', 't(VH)']]
print "P_t(nonVH < VH_t) =", (nonVH_t.trace()< VH_t.trace()).mean()

#Within group drift rate comparisons
#Control
Control_v_incongruent, Control_v_congruent = m.nodes_db.node[['v(Control.incongruent)', 'v(Control.congruent)']]
print "P_v(Control_v_incongruent > Control_v_congruent) =", (Control_v_incongruent.trace()> Control_v_congruent.trace()).mean()
Control_v_incongruent, Control_v_neutral = m.nodes_db.node[['v(Control.incongruent)', 'v(Control.neutral)']]
print "P_v(Control_v_incongruent > Control_v_neutral) =", (Control_v_incongruent.trace()> Control_v_neutral.trace()).mean()
Control_v_congruent, Control_v_neutral = m.nodes_db.node[['v(Control.congruent)', 'v(Control.neutral)']]
print "P_v(Control_v_congruent > Control_v_neutral) =", (Control_v_congruent.trace()> Control_v_neutral.trace()).mean()

#nonVH
nonVH_v_incongruent, nonVH_v_congruent = m.nodes_db.node[['v(nonVH.incongruent)', 'v(nonVH.congruent)']]
print "P_v(nonVH_v_incongruent > nonVH_v_congruent) =", (nonVH_v_incongruent.trace()> nonVH_v_congruent.trace()).mean()
nonVH_v_incongruent, nonVH_v_neutral = m.nodes_db.node[['v(nonVH.incongruent)', 'v(nonVH.neutral)']]
print "P_v(nonVH_v_incongruent > nonVH_v_neutral) =", (nonVH_v_incongruent.trace()> nonVH_v_neutral.trace()).mean()
nonVH_v_congruent, nonVH_v_neutral = m.nodes_db.node[['v(nonVH.congruent)', 'v(nonVH.neutral)']]
print "P_v(nonVH_v_congruent < nonVH_v_neutral) =", (nonVH_v_congruent.trace()< nonVH_v_neutral.trace()).mean()

#VH
VH_v_incongruent, VH_v_congruent = m.nodes_db.node[['v(VH.incongruent)', 'v(VH.congruent)']]
print "P_v(VH_v_incongruent > VH_v_congruent) =", (VH_v_incongruent.trace()> VH_v_congruent.trace()).mean()
VH_v_incongruent, VH_v_neutral = m.nodes_db.node[['v(VH.incongruent)', 'v(VH.neutral)']]
print "P_v(VH_v_incongruent > VH_v_neutral) =", (VH_v_incongruent.trace()> VH_v_neutral.trace()).mean()
VH_v_congruent, VH_v_neutral = m.nodes_db.node[['v(VH.congruent)', 'v(VH.neutral)']]
print "P_v(VH_v_congruent > VH_v_neutral) =", (VH_v_congruent.trace()> VH_v_neutral.trace()).mean()

#Within group decision boundary comparisons
#Control
Control_a_incongruent, Control_a_congruent = m.nodes_db.node[['a(Control.incongruent)', 'a(Control.congruent)']]
print "P_a(Control_a_incongruent > Control_a_congruent) =", (Control_a_incongruent.trace()> Control_a_congruent.trace()).mean()
Control_a_incongruent, Control_a_neutral = m.nodes_db.node[['a(Control.incongruent)', 'a(Control.neutral)']]
print "P_a(Control_a_incongruent > Control_a_neutral) =", (Control_a_incongruent.trace()> Control_a_neutral.trace()).mean()
Control_a_congruent, Control_a_neutral = m.nodes_db.node[['a(Control.congruent)', 'a(Control.neutral)']]
print "P_a(Control_a_congruent > Control_a_neutral) =", (Control_a_congruent.trace()> Control_a_neutral.trace()).mean()

#nonVH
nonVH_a_incongruent, nonVH_a_congruent = m.nodes_db.node[['a(nonVH.incongruent)', 'a(nonVH.congruent)']]
print "P_a(nonVH_a_incongruent > nonVH_a_congruent) =", (nonVH_a_incongruent.trace()> nonVH_a_congruent.trace()).mean()
nonVH_a_incongruent, nonVH_a_neutral = m.nodes_db.node[['a(nonVH.incongruent)', 'a(nonVH.neutral)']]
print "P_a(nonVH_a_incongruent > nonVH_a_neutral) =", (nonVH_a_incongruent.trace()> nonVH_a_neutral.trace()).mean()
nonVH_a_congruent, nonVH_a_neutral = m.nodes_db.node[['a(nonVH.congruent)', 'a(nonVH.neutral)']]
print "P_a(nonVH_a_congruent < nonVH_a_neutral) =", (nonVH_a_congruent.trace()< nonVH_a_neutral.trace()).mean()

#VH
VH_a_incongruent, VH_a_congruent = m.nodes_db.node[['a(VH.incongruent)', 'a(VH.congruent)']]
print "P_a(VH_a_incongruent > VH_a_congruent) =", (VH_a_incongruent.trace()> VH_a_congruent.trace()).mean()
VH_a_incongruent, VH_a_neutral = m.nodes_db.node[['a(VH.incongruent)', 'a(VH.neutral)']]
print "P_a(VH_a_incongruent > VH_a_neutral) =", (VH_a_incongruent.trace()> VH_a_neutral.trace()).mean()
VH_a_congruent, VH_a_neutral = m.nodes_db.node[['a(VH.congruent)', 'a(VH.neutral)']]
print "P_a(VH_a_congruent > VH_a_neutral) =", (VH_a_congruent.trace()> VH_a_neutral.trace()).mean()

