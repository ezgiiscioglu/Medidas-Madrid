#!/usr/bin/env python
# coding: utf-8

# # Diagramas de dispersión

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns #for plotting
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats

from scipy.stats import spearmanr,pointbiserialr # for Spearman Correlation and Biserial

import statsmodels.api as sm
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from sklearn.metrics import confusion_matrix
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import anderson, norm, uniform
from skgof import cvm_test, ks_test
import statsmodels.stats.diagnostic as sm_diagnostic

from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


data = pd.read_csv("dfMedidas1.csv")
df = data.copy()


# In[3]:


df.head()


# In[38]:


xi = df.T_MAX
y = df.CO
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
line = slope*xi+intercept

plt.plot(xi,y,'o', xi, line)
plt.xlabel("T max ºC")
plt.ylabel("CO")
plt.title('Diagrama de Dispersión')
ax = plt.gca()
fig = plt.gcf()

# plot(T_MAX, CO, main="Diagrama de Dispersión", xlab="T max ºC", ylab="CO", pch=19)
# abline(lm(CO ~ T_MAX), col = "red") # regresión (y~x)

# I couldn't find smoothScatter for python


# In[42]:


xi = df.T_MAX
y = df.O3
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
line = slope*xi+intercept

plt.plot(xi,y,'o', xi, line)
plt.xlabel("T max ºC")
plt.ylabel("O3")
plt.title('Diagrama de Dispersión')
ax = plt.gca()
fig = plt.gcf()

# plot(T_MAX, O3, main="Diagrama de Dispersión", xlab="T max ºC", ylab="O3", pch=19)
# abline(lm(O3 ~ T_MAX), col = "red") # regresión (y~x)

# I couldn't find smoothScatter for python


# In[47]:


df.corr()


# In[5]:


# pip install plotly --upgrade


# In[4]:


# pip install chart_studio


# In[ ]:


import plotly
import plotly.tools as tls
import chart_studio.plotly as py

plt.figure(figsize=(12,8))
plt.scatter(x=df.T_MAX, y=df.CO)
plt.xlabel("T max ºC")
plt.ylabel("CO")
plt.title('Diagrama de Dispersión')
fig = plt.gcf()

plotly_fig = tls.mpl_to_plotly(fig)
py.iplot(plotly_fig, filename = 'mpl-scatter-line')
plt.show()


library(car)
scatterplot(O3 ~ T_MAX, data = dfMedidas, main ="Diagrama de Dispersión", 
            xlab ="T max ºC", ylab ="O3", lwd = 2, lty= 2, col = "green", cex = 0.5,
            ellipse = list(levels = c(.5, .95), robust = TRUE, fill = TRUE, fill.alpha = 0.1),  
            regLine = list(method = lm, lty = 1, lwd = 2, col = "blue"),
            smooth = list(smoother = loessLine, col.spread = "red", col.var = "red", lty.var = 2, lty.var = 4)
           )
bu kısım daha yapılmadı


# In[70]:


plt.figure(figsize=(8, 8))

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)

# the scatter plot:
ax_scatter.scatter(df.T_MAX, df.O3)

ax_histx.hist(df.T_MAX)
ax_histy.hist(df.O3, orientation='horizontal')

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())

ax_scatter.set_xlabel('T max ºC')
ax_scatter.set_ylabel('O3')

ax_histx.set_xlabel('Marginal x label')
ax_histy.set_ylabel('Marginal y label')


plt.show()

# scatter.hist(T_MAX, O3, main ="Diagrama de Dispersión", xlab ="T max ºC", ylab ="O3")
#Igual que la anterior pero con histogramas y coeficiente de regresión


# In[4]:


df["Lluvia_SN"] = np.where(df['Lluvia'] > 0, 'Si','No')
df["PM2_half_UMBRAL"] = np.where(df['PM2.5'] >= 25, 'Si','No')
df["PM10_UMBRAL"] = np.where(df['PM10'] >= 50, 'Si','No')
df["T_MAX_ALERTA"] = np.where(df['T_MAX'] >= 36, 'A','V')
df["T_MAX_ALERTA"] = np.where(df['T_MAX'] >= 39, 'N', df["T_MAX_ALERTA"])
df["T_MAX_ALERTA"] = np.where(df['T_MAX'] >= 42, 'R', df["T_MAX_ALERTA"])


# In[80]:


plt.figure(figsize=(12,8))
plt.scatter(x=df.T_MAX[df.Lluvia_SN=='Si'], y=df.O3[(df.Lluvia_SN=='Si')], c="red")
plt.scatter(x=df.T_MAX[df.Lluvia_SN=='No'], y=df.O3[(df.Lluvia_SN=='No')])
plt.legend(["Si", "No"])
plt.xlabel("T max ºC")
plt.ylabel("O3")
plt.show()

# scatterplot(O3 ~ T_MAX |Lluvia_SN, data = dfMedidas, main ="Diagrama de Dispersión", xlab ="T max ºC", ylab ="O3",smoother = TRUE)


# In[83]:


sns.lmplot(x="T_MAX", y="O3", hue="Lluvia_SN", data=df, markers=["o", "x"]);
# Visualizing linear relationships


# In[91]:


sns.pairplot(df, diag_kind="kde", kind="reg", vars=["O3","SO2","NO","Viento_MED","T_MAX"])

#scatterplotMatrix(~ O3 + SO2 + NO + Viento_MED + T_MAX, data = dfMedidas,
#                  smooth = list(smoother = loessLine, spread = TRUE, lty.smooth = 1, 
#                                lwd.smooth = 1.5, lty.spread = 3, lwd.spread = 1),
#                  ellipse = TRUE, cex = 0.25, col = "red", 
#                  regLine = list(method=lm, lty=1, lwd=2, col="blue")) 


# In[101]:


from scipy.stats import pearsonr

def corrfunc(x,y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    # Unicode for lowercase rho (ρ)
    rho = '\u03C1'
    ax.annotate(f'{rho} = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    
g = sns.pairplot(df, vars=["O3","SO2","NO","Viento_MED","T_MAX"])
g.map_lower(corrfunc)
plt.show()

# As far as I understand there is no function to see chart correlation in scatter but I'm not really sure
# So, I searched and I found this function

# library(PerformanceAnalytics)
# chart.Correlation(dfMedidas[,c("O3", "SO2", "NO", "Viento_MED", "T_MAX")])


# ## En ºC Covariance

# In[103]:


df.T_MIN.cov(df.CO)
# cov(T_MIN, CO)


# In[104]:


df.T_MAX.cov(df.CO)
# cov(T_MAX, CO)


# In[105]:


df.T_MIN.cov(df.O3)
# cov(T_MIN, O3)


# In[106]:


df.T_MAX.cov(df.O3)
# cov(T_MAX, O3)


# In[107]:


# T_MAX ~ O3


# ## En ºF Covariance

# In[109]:


((df.T_MIN* 9/5)+32).cov(df.CO)
# cov((T_MIN * 9/5)+32, CO)


# In[110]:


((df.T_MAX* 9/5)+32).cov(df.CO)
# cov((T_MAX * 9/5)+32, CO)


# In[111]:


((df.T_MIN* 9/5)+32).cov(df.O3)
# cov((T_MIN * 9/5)+32, O3)


# In[112]:


((df.T_MAX* 9/5)+32).cov(df.O3)
# cov((T_MAX * 9/5)+32, O3)


# ## En ºC Correlation

# In[117]:


df.T_MIN.corr(df.CO)
# cor(T_MIN, CO)


# In[118]:


df.T_MAX.corr(df.CO)
# cor(T_MAX, CO)


# In[119]:


df.T_MIN.corr(df.O3)
# cor(T_MIN, O3)


# In[120]:


df.T_MAX.corr(df.O3)
# cor(T_MAX, O3)


# ## En ºF Correlation

# In[123]:


((df.T_MIN* 9/5)+32).corr(df.CO)
# cor((T_MIN * 9/5)+32, CO)


# In[122]:


((df.T_MAX* 9/5)+32).corr(df.CO)
# cor((T_MAX * 9/5)+32, CO)


# In[124]:


((df.T_MIN* 9/5)+32).corr(df.O3)
# cor((T_MIN * 9/5)+32, O3)


# In[125]:


((df.T_MAX* 9/5)+32).corr(df.O3)
# cor((T_MAX * 9/5)+32, O3)


# In[ ]:


# Pearson's correlation coefficient is bounded in the interval [-1,1], is dimensionless and has the sign of covariance.
# El coeficiente de correlación de Pearson está acotado en el intervalo [-1,1], es adimensional y tiene el signo de la covarianza.


# In[126]:


df.CO.corr(df.SO2)
# cor(CO,SO2)


# In[129]:


spearmanr(df.O3,df.T_MAX)
# cor.test(O3,T_MIN, method = "spearman")


# In[133]:


spearmanr(df.O3,df.T_MIN)
# cor.test(O3,T_MIN, method = "spearman")


# In[134]:


spearmanr(df.CO,df.T_MAX)
# cor(CO,T_MAX, method = "spearman")


# In[135]:


spearmanr(df.CO,df.T_MIN)
# cor.test(CO,T_MIN, method = "spearman")


# In[ ]:


# Comparing with Pearson's correlation value, a possible inverse non-linear relationship between carbon monoxide and temperature is revealed, even higher with the minimum temperature.


# In[5]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['Lluvia_SN'] = le.fit_transform(df['Lluvia_SN'])
df['PM2_half_UMBRAL'] = le.fit_transform(df['PM2_half_UMBRAL'])
df['PM10_UMBRAL'] = le.fit_transform(df['PM10_UMBRAL'])
df['T_MAX_ALERTA'] = le.fit_transform(df['T_MAX_ALERTA'])


# In[150]:


pointbiserialr(df.CO, df.Lluvia_SN)
# biserial.cor(CO, Lluvia_SN)


# In[151]:


pointbiserialr(df.O3, df.Lluvia_SN)
# biserial.cor(O3, Lluvia_SN)


# In[ ]:


# In Python it is used pointbiserialr for biserial but I didn't understand why the result is reverse


# #  Modelos de predicción

# ## Regresión lineal

# In[6]:


df.Mes = pd.Categorical(df.Mes)
df.Dia_sem = pd.Categorical(df.Dia_sem)
df.Fecha = pd.Categorical(df.Fecha)

df['Mes'] = le.fit_transform(df['Mes'])
df['Dia_sem'] = le.fit_transform(df['Dia_sem'])
df['Fecha'] = le.fit_transform(df['Fecha'])


# In[7]:


# calculate the correlation matrix
corr = df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True)
# heatmap(h$correlations, Rowv = NA, Colv = NA, col = heat.colors(256), na.rm=TRUE)


# In[16]:


# calculate the correlation matrix
corr = df.corr()
f, ax2 = plt.subplots(1, figsize=(24,20))

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2,
        square=True)
# library(gplots)
# mi_color <- colorRampPalette(c("red", "black", "red"))(n = 299)
# heatmap.2(h$correlations,
          #cellnote = h$correlations,  # same data set for cell labels
          #dendrogram = "none", 
          # col=mi_color,
          # Rowv = FALSE, 
          # notecol="black",      # change font color of cell labels to black
          # density.info="none",  # turns off density plot inside color legend
          # trace="none",         # turns off trace lines inside the heat map
          # margins =c(8,8),     # widens margins around plot
          # dendrogram = "none",
          # Colv="NA")            # turn off column clustering


# In[21]:


round(h$correlations[order(abs(h$correlations[,"NO2"]),decreasing = TRUE),"NO2"],2)

I couldn't do this!


# In[13]:


matplotlib.style.use('ggplot')

plt.scatter(df.CO, df.NO2)
plt.xlabel("Concentración CO")
plt.ylabel("Concentración NO2")
plt.show()

# library("ggpubr")
# ggscatter(dfMedidas, x = "CO", y = "NO2", 
#          add = "reg.line", conf.int = TRUE, 
#          cor.coef = TRUE, cor.method = "pearson",
#          xlab = "Concentración CO", ylab = "Concentración NO2")


# In[11]:


round(stats.pearsonr(df['CO'], df['NO2'])[0], 2)


# In[22]:


# Statsmodel
X = df["CO"]
y = df["NO2"]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()

# l1 <-lm(NO2 ~ CO, data = dfMedidas)
# summary(l1)


# In[99]:


X = df["Viento_MAX"]
y = df["NO2"]
X = sm.add_constant(X)


fit1 = sm.OLS(y, X).fit()
predictions = fit1.predict(X)
fit1.summary()

X2 = df[["Viento_MAX", "T_MAX"]]
y2 = df["NO2"]
X2 = sm.add_constant(X2)

fit2 = sm.OLS(y2, X2).fit()
predictions = fit2.predict(X2)
fit2.summary()

X = df[["Viento_MAX", "T_MAX","Lluvia"]]
y = df["NO2"]
X = sm.add_constant(X)

fit3 = sm.OLS(y, X).fit()
predictions = fit3.predict(X)
fit3.summary()

fit1.summary()

# fit1 <- lm(NO2 ~ Viento_MAX , data = dfMedidas)
# fit2 <- lm(NO2 ~ Viento_MAX + T_MAX, data = dfMedidas)
# fit3 <- lm(NO2 ~ Viento_MAX + T_MAX + Lluvia, data = dfMedidas)
# fit1


# In[61]:


import statsmodels.formula.api as smf
lm = smf.ols("NO2 ~ Viento_MAX", df)
fit1 = lm.fit()
fit1.summary()

# The same with the top


# In[26]:


fit1.params
# fit1


# In[29]:


fit2.summary()


# In[30]:


fit2.params
# fit2


# In[31]:


fit3.summary()


# In[32]:


fit3.params
# fit3


# In[33]:


fit1.summary().tables[1]
# coefficients(fit1) # coeficientes del modelo


# In[36]:


fit1.conf_int()


# In[74]:


fit1.fittedvalues[0:10]
# head(fitted(fit1),10) # valores predichos


# In[75]:


y[0:10]
# real values


# In[124]:


fit1.resid[0:10]
# head(residuals(fit1),10) # residuos


# In[62]:


residuos = pd.DataFrame({"fit1real_y" : y[0:10],
                   "fit1pred_y" : fit1.fittedvalues[0:10]})
residuos


# In[63]:


residuos["error"] = residuos["fit1real_y"] - residuos["fit1pred_y"]
# The same but this one is longer


# In[64]:


residuos


# In[97]:


sm.stats.anova_lm(fit1)
# anova(fit1) # tabla anova  


# In[105]:


# vcov(fit1) # matriz de covarianza
# Normally in python we use np.cov() but I couldn't solve


# In[108]:


sm.stats.anova_lm(fit1,fit2,fit3)


# In[127]:


X = df[["Viento_MAX", "T_MAX","Lluvia_SN"]]
y = df["NO2"]
X = sm.add_constant(X)

fit4 = sm.OLS(y, X).fit()
predictions = fit4.predict(X)

sm.stats.anova_lm(fit1,fit2,fit4)

# fit4 <- lm(NO2 ~ Viento_MAX + T_MAX + Lluvia_SN, data = dfMedidas)
# anova(fit1,fit2,fit4)


# In[128]:


fit4.summary()
# step(lm(NO2 ~ Viento_MAX + T_MAX + Lluvia_SN, data = dfMedidas), direction = "both")


# In[23]:


V_max = np.arange(df.Viento_MAX.min(),df.Viento_MAX.max(),step=20)
T_max = np.arange(df.T_MAX.min(),df.T_MAX.max(),step=20)

def z(x,y):
    return (x*(-1.0428) + y*(-0.5581) + (86.1814 ))

NO2_pred = np.outer(V_max, T_max, z)


# In[ ]:


trace1 = go.Scatter3d(
    x=dataframe.world_rank,
    y=dataframe.research,
    z=dataframe.citations,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Olmadı


# ## Selección de atributos

# In[15]:


X = df[["Viento_MAX", "T_MAX","Lluvia","Lluvia_SN"]]
y = df["NO2"]
X = sm.add_constant(X)

fit5 = sm.OLS(y, X).fit()
predictions = fit5.predict(X)


# In[16]:


# As I understand there is no function to feature importance for linear regression


# ## Regresión no Lineal

# In[21]:


matplotlib.style.use('ggplot')

plt.scatter(df.Viento_MAX, df.NO2,s=20)
plt.xlabel("Viento_MAX")
plt.ylabel("NO2")
plt.show()

# ggplot(dfMedidas, aes(x = Viento_MAX, y = NO2)) + geom_point(size=0.5)


# In[22]:


# In python there is no function to apply directly nonlinear regression
# It can do least-squares curve fitting, but it only provides you with parameter estimates
# I couldn't convert them


# ## Regresión de atributos no continuos

# In[69]:


name = ['BP', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(fit2.resid, fit2.model.exog)
lzip(name, test)

# library(lmtest)
# bptest(fit2) #test Breush-Pagan


# In[79]:


from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(fit2.resid)
# dwtest(fit2) #test Durbin-Watson


# In[83]:


dw


# In[87]:


plot_acf(fit2.resid)
plt.show()
# acf(fit2$residuals)


# In[ ]:


# I couldn't find a equivalent function hetcor


# In[100]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif["features"] = X2.columns

# vif(fit2)


# In[101]:


vif.round(10)


# In[102]:


df.Viento_MAX.var()
# var(Viento_MAX)


# In[104]:


df.T_MAX.var()
# var(T_MAX)


# In[110]:


anderson(fit2.resid)
# library(nortest)
# ad.test(fit2$residuals)


# In[113]:


# pip install scikit-gof


# In[123]:


cvm_test(fit2.resid, uniform(0, 5))
# cvm.test(fit2$residuals)


# In[137]:


print(sm_diagnostic.kstest_normal(x = fit2.resid, dist = "norm"))
# lillie.test(fit2$residuals)


# In[140]:


# With gvlma function you can integrate them all


# ## Modelos lineales generalizados

# In[12]:


X = df[["Viento_MAX", "T_MAX"]]
y = df["NO2"]
X = sm.add_constant(X)

fit_glm = sm.GLM(y, X, family=sm.families.Gaussian()).fit()

# fit_glm <- glm(NO2 ~ Viento_MAX + T_MAX, data = dfMedidas, family = gaussian)
# fit_glm


# In[13]:


fit_glm


# In[14]:


fit_glm.summary()


# In[20]:


fit_glm2 = sm.GLM(y, X, family=sm.families.Gaussian(sm.families.links.log)).fit()

# fit_glm2 <- glm(NO2 ~ Viento_MAX + T_MAX, data = dfMedidas, family = gaussian(link = "log"))
# fit_glm2 


# In[21]:


fit_glm2.summary()


# In[20]:


X = df[["Viento_MAX", "T_MAX"]]
y = df["PM2_half_UMBRAL"]
X = sm.add_constant(X)

fit_log = sm.GLM(y, X, family=sm.families.Binomial())
fit_log_fit = fit_log.fit()
# fit_log <- glm(PM2.5_UMBRAL ~ Viento_MAX + T_MAX, data = dfMedidas, family = binomial)
# summary(fit_log) 


# In[21]:


fit_log_fit.summary()


# In[ ]:


# Didn't finish


# In[ ]:





# In[26]:


X = df[["Viento_MAX", "T_MAX", "Lluvia_SN"]]
y = df["PM2.5_MAX"]
X = sm.add_constant(X)

fit_poi = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# fit_poi <- glm(PM2.5_MAX ~ Viento_MAX + T_MAX + Lluvia_SN, data = dfMedidas, family = poisson)
# summary(fit_poi) 


# In[27]:


fit_poi.summary()


# In[ ]:





# In[30]:


X = df[["Viento_MAX", "Lluvia_SN"]]
y = df["PM2.5_MAX"]
X = sm.add_constant(X)

fit_poi2 = sm.GLM(y, X, family=sm.families.Poisson()).fit()
# fit_poi2 <- glm(PM2.5_MAX ~ Viento_MAX + Lluvia_SN, data = dfMedidas, family = poisson)
# summary(fit_poi2) 


# In[31]:


fit_poi2.summary()


# In[ ]:




