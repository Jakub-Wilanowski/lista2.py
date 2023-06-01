import ssl
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score


ssl._create_default_https_context = ssl._create_unverified_context

f = open("pliktextowy.txt", "r")

url = f.readline()
headers = f.readlines()
df = pd.read_csv(url,names=headers)


#df = pd.DataFrame() # tutaj podmień df. Ma zawierać wczytane dane.

#Zadanie1 przypisz nazwy kolumn z df w jednej linii:   (2pkt)

wynik1 = df.keys()
print(wynik1)

#Zadanie 2: Wypisz liczbę wierszy oraz kolumn ramki danych w jednej linii.  (2pkt)
wynik2 = df.shape
print(wynik2)


#Zadanie Utwórz klasę Wine na podstawie wczytanego zbioru:
#wszystkie zmienne objaśniające powinny być w liscie.
#Zmienna objaśniana jako odrębne pole.
# metoda __init__ powinna posiadać 2 parametry:
#listę (zmienne objaśniające) oraz liczbę(zmienna objaśniana).
#nazwy mogą być dowolne.

class Wine:
    def __init__(self, listaObjaśniające, answ):
        self.list = listaObjaśniające
        self.num =answ
    def __repr__(self):
        return "Wine("+str(self.list)+" , "+str(self.num)+")"
# Klasa powinna umożliwiać stworzenie nowego obiektu na podstawie
# już istniejącego obiektu jak w pdf z lekcji lab6.
# podpowiedź: metoda magiczna __repr__
#Nie pisz metody __str__.

#Zadanie 3 Utwórz przykładowy obiekt:   (3pkt)
wynik3 = Wine([1999.23,1.791,2.43,15.6,2137,2.8,3.46,.28,2.29,2.62,1.04,3.32,1065],1) #do podmiany. Pamiętaj - ilość elementów, jak w zbiorze danych.
#Uwaga! Pamiętaj, która zmienna jest zmienną objaśnianą
print(wynik3)

#Zadanie 4.                             (3pkt)
#Zapisz wszystkie dane z ramki danych do listy obiektów typu Wine.
#Nie podmieniaj listy, dodawaj elementy.
##Uwaga! zobacz w jakiej kolejności podawane są zmienne objaśniające i objaśniana.
# Podpowiedź zobacz w pliktextowy.txt

wineList = []
for line in df.values.tolist():
    wineList.append(Wine(answ=line.pop(0),listaObjaśniające=line))
wynik4 = len(wineList)
print(wynik4)


#Zadanie5 - Weź ostatni element z listy i na podstawie         (3pkt)
#wyniku funkcji repr utwórz nowy obiekt - eval(repr(obiekt))
#do wyniku przypisz zmienną objaśnianą z tego obiektu:
wynik5 = eval(repr(wineList[-1])).num
print(wynik5)


#Zadanie 6:                                                          (3pkt)
#Zapisz ramkę danych  do bazy SQLite nazwa bazy(dopisz swoje imię i nazwisko):
# wines_imie_nazwisko, nazwa tabeli: wines.
#Następnie wczytaj dane z tabeli wybierając z bazy danych tylko wiersze z typem wina nr 3
# i zapisz je do nowego data frame:
wynik6 = "W następnej linijce podmień na nowy  data frame z winami tylko klasy trzeciej:"
wynik6 = pd.DataFrame() #tutaj do podmiany

print(wynik6.shape)


#Zadanie 7                                                          (1pkt)
#Utwórz model regresji Logistycznej z domyślnymi ustawieniami:

model = LogisticRegression()


wynik7 = model.__class__.__name__
print(wynik7)

# Zadanie 8:                                                        (3pkt)
#Dokonaj podziału ramki danych na dane objaśniające i  do klasyfikacji.
#Znormalizuj dane objaśniające za pomocą:
#X = preprocessing.normalize(X)
# Wytenuj model na wszystkich danych bez podziału na zbiór treningowy i testowy.
# Wykonaj sprawdzian krzyżowy, używając LeaveOneOut() zamiast KFold (Parametr cv)
#  Podaj średnią dokładność (accuracy)
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
x = preprocessing.normalize(x)
model.fit(x,y)
tempList = cross_val_score(estimator=model,X=x,y=y, cv=LeaveOneOut(), scoring= "accuracy")
wynik8 = tempList.mean()
print(wynik8)