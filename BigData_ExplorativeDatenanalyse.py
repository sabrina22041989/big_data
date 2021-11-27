#!/usr/bin/env python
# coding: utf-8

# # Spotify - Deskriptive Analyse

# In[1]:


pip install pyspark


# In[2]:


pip install pandas


# In[3]:


pip install seaborn


# In[4]:


pip install pyarrow


# In[5]:


pip install pyspark


# In[6]:


#Import notwendiger Libraries
from pyspark.sql import SQLContext, SparkSession 
from pyspark.sql.types import StructType, DateType, StringType, IntegerType, FloatType
from pyspark.sql.types import *
import numpy as np
import pandas as pd
#from tqdm.notebook import tqdm
import ast
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.functions import min, max
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import udf, concat, col, lit
import re
from spark_plot import mpl
from collections import Counter
import pyarrow as pa
from pyspark.sql.functions import col, skewness, kurtosis
import matplotlib.pyplot as plt
import seaborn as sb
from spark_plot import mpl


# In[7]:


#Erstellung einer Spark-Session
spark = SparkSession.builder.appName('bigdata_spark').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(spark.sparkContext)


# In[8]:


#Import des Spotify-Datensatzes aus Hadoop
spotify = sc.textFile('hdfs://localhost:9000/input/Spotify.txt')


# In[9]:


#Erstellung von Dataframes und dazugehörigen Datentypen
spotify_schema = StructType()
spotify_schema.add("Rank",IntegerType(),True)
spotify_schema.add("Track",StringType(),True)
spotify_schema.add("Artist",StringType(),True)
spotify_schema.add("Streams",IntegerType(),True)
spotify_schema.add("Link",StringType(),True)
spotify_schema.add("Week",DateType(),True)
spotify_schema.add("Album_Name",StringType(),True)
spotify_schema.add("Duration_MS",IntegerType(),True)
spotify_schema.add("Explicit",StringType(),True)
spotify_schema.add("Track_Number_on_Album",IntegerType(),True)
spotify_schema.add("Artist_Followers",IntegerType(),True)
spotify_schema.add("Artist_Genres",StringType(),True)

df_spotify = sqlContext.read.options(delimiter=',').schema(spotify_schema).csv('hdfs://localhost:9000/input/Spotify.txt', header=True)


# In[10]:


#Splitten des Attributs "Woche" in Year, Month, Day
df_spotify_split_week = df_spotify.withColumn('Year', split(df_spotify['Week'], '-').getItem(0))        .withColumn('Month', split(df_spotify['Week'], '-').getItem(1))        .withColumn('Day', split(df_spotify['Week'], '-').getItem(2))
df_spotify_split_week.printSchema()


# In[11]:


#Anzahl der Datensätze
print('Anzahl der Datensätze:')
print(df_spotify.count(),'\n')


# In[12]:


#Anzahl der Spalten
print('Anzahl der Spalten:')
print(len(df_spotify.columns),'\n')


# In[13]:


#Nullwerte rausfiltern
df_spotify_null = df_spotify_split_week.select([count(when(col(c).isNull(), c)).alias(c) for c in df_spotify_split_week.columns]).toPandas().T
print("Anzahl von Nullwerten pro Attribut:",df_spotify_null,'\n')


# In[14]:


#Entfernen der Nullwerte
df_spotify_Nullwerte_entfernt = df_spotify_split_week.na.drop(how="any",thresh=None, subset=None)
df_spotify_ohne_Nullwerte = df_spotify_Nullwerte_entfernt.select([count(when(col(c).isNull(), c)).alias(c) for c in df_spotify_Nullwerte_entfernt.columns]).toPandas().T
print("Datensatz ohne Nullwerte",df_spotify_ohne_Nullwerte,'\n')


# Datenaufbereitung

# In[15]:


#Duplikate untersuchen
num = df_spotify_Nullwerte_entfernt.count()
uniq = df_spotify_Nullwerte_entfernt.distinct().count()

if num > uniq:
    print('Datensatz hat Duplikate')
else:
    print('Keine Duplikate')


# Statistische Analyse

# In[17]:


#Statistische Analyse
num_cols = ['Rank','Streams','Duration_MS','Artist_Followers']
df_spotify_Nullwerte_entfernt.select(num_cols).describe().show()

def describe_pd(df_spotify_Nullwerte_entfernt, columns, deciles=False):
    
    if deciles:
        percentiles = np.array(range(0, 110, 10))
    else:
        percentiles = [25, 50, 75]

    percs = np.transpose([np.percentile(df_spotify_Nullwerte_entfernt.select(x).collect(), percentiles) for x in columns])
    percs = pd.DataFrame(percs, columns=columns)
    percs['summary'] = [str(p) + '%' for p in percentiles]

    spark_describe = df_spotify_Nullwerte_entfernt.describe().toPandas()
    new_df = pd.concat([spark_describe, percs],ignore_index=True)
    new_df = new_df.round(2)
    return new_df[['summary'] + columns]

print(describe_pd(df_spotify_Nullwerte_entfernt,num_cols))


# In[18]:


#Skewness & Kurtosis
var = 'Rank'
df_spotify_Nullwerte_entfernt.select(skewness(var), kurtosis(var)).show()

var_1 = 'Streams'
df_spotify_Nullwerte_entfernt.select(skewness(var_1), kurtosis(var_1)).show()

var_2 = 'Duration_MS'
df_spotify_Nullwerte_entfernt.select(skewness(var_2), kurtosis(var_2)).show()

var_3 = 'Artist_Followers'
df_spotify_Nullwerte_entfernt.select(skewness(var_3), kurtosis(var_3)).show()


# In[87]:


print('Deskriptive Analyse\n')
print('Ziel: In welche Künstler/-innen lohnt es sich aus Sicht von Musikprodzent/-innen, Konzertveranstalter/-innen und anderern Akteur/-innen der Musikindustrie zu investieren?\n')

#Erstellung einer Tabelle für den späteren Einsatz von SparkSQL
df_spotify_Nullwerte_entfernt.registerTempTable('Spotify')
sqlContext = SQLContext(spark)


# In[42]:


print('Frage 1: Aus wie vielen Künstler/-innen, Songs und Musikgattungen besteht der Datensatz?\n')
df1 = sqlContext.sql('SELECT COUNT (DISTINCT Artist) AS Kuenstler, COUNT (DISTINCT Track) AS Songs, COUNT (DISTINCT Artist_Genres) AS Musikgattungen FROM Spotify').toPandas()
print(df1)
#df1.to_csv('/home/bigdata/Dokumente/anzahl_kuenstler_lieder_genres.csv')


# In[43]:


print('\nFrage 2: Wie viele Künstler/-innen, Songs und Musikgattungen waren pro Jahr in den Charts?\n')
df2 = sqlContext.sql('SELECT Year AS Jahr, COUNT (DISTINCT Artist) AS Kuenstler, COUNT (DISTINCT Track) AS Songs, COUNT (DISTINCT Artist_Genres) AS Musikgattungen FROM Spotify GROUP BY Year ORDER BY Year').toPandas()
print(df2)
#df2.to_csv('/home/bigdata/Dokumente/anzahl_kuenstler_lieder_genres_pro_jahr.csv')


# In[44]:


print('\nFrage 3: Welche Künstler/-innen kommen, gemessen an der Anzahl der Wochen, am häufigsten in den Top 200 Charts vor?')
df3 = sqlContext.sql('SELECT Artist, COUNT (Artist) AS Haeufigkeit FROM Spotify GROUP BY Artist ORDER BY COUNT (Artist) DESC').toPandas()
print(df3.head(21))
#df3.to_csv('/home/bigdata/Dokumente/anzahl_kuenstler_charts.csv')


# In[47]:


print('\nFrage 4: Welche Künstler/-innen hatten zwischen 2017 und 2021 die meisten Follower?')
df4 = sqlContext.sql('SELECT Artist AS Kuenstler, MAX(Artist_Followers) AS Follower FROM Spotify GROUP BY Kuenstler ORDER BY MAX(Artist_Followers) DESC').toPandas()
print(df4.head(21))
#df4.to_csv('/home/bigdata/Dokumente/anzahl_follower_pro_kuenstler.csv')


# In[96]:


print('\nFrage 5: Welche Künstler/-innen erzielten die meisten Streams pro Track bezogen auf die einzelnen Jahre?')
df56 = sqlContext.sql('SELECT Artist AS Kuenstler, MAX(Streams) AS Streams, Track AS Song, Year AS Jahr FROM Spotify GROUP BY Kuenstler, Track, Jahr ORDER BY Streams DESC').toPandas()
print(df56.head(21))
#df56.to_csv('/home/bigdata/Dokumente/anzahl_streams_pro_track_jahre.csv')


# In[93]:


print('\nFrage 6: Welche Künstler/-innen waren am häufigsten, gemessen an der Anzahl der Wochen, auf den ersten 10 Plätzen der Charts?')
df7 = sqlContext.sql('SELECT Artist AS Kuenstler, COUNT(Artist) AS Wochen, Rank FROM Spotify where Rank <= 10 GROUP BY Kuenstler, Rank ORDER BY Wochen DESC').toPandas()
print(df7.head(21))
#df7.to_csv('/home/bigdata/Dokumente/kuenstler_am_laengsten_in_charts.csv')


# In[95]:


print('\nFrage 7: Welche Songs hatten die meisten Streams?')
df5 = sqlContext.sql('SELECT Track AS Song, MAX(Streams) AS Streams FROM Spotify GROUP BY Song ORDER BY Streams DESC').toPandas()
print(df5.head(21))
#df5.to_csv('/home/bigdata/Dokumente/streams_songs.csv')


# In[97]:


print('\nFrage 8: Welche Musikgattung hatte die meisten Streams?')
df12 = sqlContext.sql('SELECT distinct Artist_Genres AS Musikgattung, Streams FROM Spotify ORDER BY Streams DESC').toPandas()
print(df12.head(21))
#df12.to_csv('/home/bigdata/Dokumente/streams_genre.csv')


# In[98]:


print('\nFrage 9: Wie lange dauerten Songs an, die in den Top 10 Charts waren?')
df13 = df_spotify_Nullwerte_entfernt.groupby('Rank','Track').avg('Duration_MS').sort(col('Rank').asc()).where(col('Rank') <= 10).toPandas()
print(df13.head(21))
#df13.to_csv('/home/bigdata/Dokumente/durchschnittliche_laenge.csv')


# In[84]:


print('\nFrage 10: Welche Musikgattungen wurden pro Jahrezeit am meisten gehört?')
df9 = sqlContext.sql('SELECT Artist_Genres AS Musikgattung, Month AS Monat FROM Spotify ORDER BY Monat').toPandas()
df10 = df9[['Musikgattung','Monat']].value_counts()
print(df10.head(21))
#df10.to_csv('/home/bigdata/Dokumente/genre_jahreszeiten.csv')


# In[30]:


print('\nFrage 11: Wieviel Lieder gab es pro Musikgattung?')
df8 = sqlContext.sql('SELECT Artist_Genres AS Musikgattung, COUNT(distinct Track) AS Song FROM Spotify GROUP BY Artist_Genres ORDER BY COUNT(distinct Track) DESC').toPandas()
print(df8.head(21))
#df8.to_csv('/home/bigdata/Dokumente/anzahl_lieder_genre.csv')


# In[99]:


print('\nFrage 12: Welche Künstler/-innen erzielten die meisten Streams?')
df6 = sqlContext.sql('SELECT Artist AS Kuenstler, MAX(Streams) AS Streams FROM Spotify GROUP BY Artist ORDER BY MAX(Streams) DESC').toPandas()
print(df6.head(21))
#df6.to_csv('/home/bigdata/Dokumente/streams_kuenstler.csv')


# In[100]:


pandasDF = df_spotify_Nullwerte_entfernt.toPandas()
print('\nFrage 13: Welche Künstler/-innen sind der Musikgattung "PoP, Pop-Teen pop" zuzuordnen?')
l= pandasDF[pandasDF['Artist_Genres']=="['pop', 'post-teen pop']"]['Artist'].value_counts()
print(l)
#l.to_csv('/home/bigdata/Dokumente/genre_pop_post-teen_pop.csv')


# In[38]:


#Korrelationsmatrix:
corr_matrix = pandasDF.corr()
corr_matrix
sb.heatmap(corr_matrix,annot=True)


# In[39]:


df20 = df_spotify_Nullwerte_entfernt.toPandas()
df20.to_csv('/home/bigdata/Dokumente/komplette_dateien.csv')


# In[ ]:





# In[24]:


#Graph 3:
pandasDF['Artist'].value_counts().head(10).plot.bar(figsize=(20,10))
plt.xlabel('Künstler')
plt.ylabel('Anzahl Tracks')
plt.title('Top 10 Künstler')


# In[22]:


#Graph 1:
#print('\nAnzahl der Lieder pro Jahr in Prozent')
pandasDF = df_spotify_Nullwerte_entfernt.toPandas()
pandasDF['Year'].value_counts().plot.pie(figsize=(10,10),autopct='%1.1f%%')
plt.title('Anzahl der Lieder pro Jahr in Prozent')
plt.show()


# In[36]:


#Graph 2:
pandasDF[pandasDF['Artist_Genres']=="['pop', 'post-teen pop']"]['Artist'].value_counts().plot.pie(figsize=(10,10),autopct='%1.1f%%')
plt.title('Anteil der Songs aus dem Genre "pop, post-teen pop" pro Künstler')


# In[31]:


#Graph 5:
pandasDF = df_spotify_Nullwerte_entfernt.toPandas()
pandasDF['Artist_Genres'].value_counts().head(10).plot.bar(figsize=(20,10))
plt.xlabel('Top genre Name')
plt.ylabel('Number of song')
plt.title('Top 10 genre bar plot') 


# In[29]:


#Graph 4:
pandasDF['Artist'].value_counts().head(10).plot.pie(figsize=(10,10), autopct='%1.0f%%')
plt.title('Top 10 Künstler basierend auf der Anzahl an Songs in Prozent')

