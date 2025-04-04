# %%
import streamlit as st


st.set_page_config(layout='wide')

import country_converter as coco
import numpy as np
import plotly.graph_objects as go
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import pandas as pd
import json
from streamlit_folium import st_folium
import re

# !pip install folium
# !pip install geopandas
import geopandas as gpd
import folium

# zorgen dat mijn kolommen niet gelimiteerd zijn
pd.set_option('display.max_columns', None)

# url van de api in een cache zetten
url = "https://beer9.p.rapidapi.com/"

# uit de database aantal bieren opvragen
querystring = {"limit": 45000}

headers = {
	"x-rapidapi-key": "4e798ca430msh319c1c6e9f29f7dp1b12b7jsn5381c0e8f29b",
	"x-rapidapi-host": "beer9.p.rapidapi.com"
}

# response = requests.get(url, headers=headers, params=querystring)

# if response.status_code == 200:
#     data = response.json()  # Converteer API-response naar een Python dictionary
#     print(data.keys())  # Check de structuur


# df = pd.DataFrame(data["data"])

@st.cache_data
def load_data(url):
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        data = response.json()
        print(data.keys())
    return pd.DataFrame(data["data"])  # Geef volledige dataframe terug

df = load_data(url)  # Dit is nu je volledige dataset
amerikaansbier = df[df['country'] == 'United States']  # Amerikaanse subset


# Bekijken hoeveel biertjes er per regio zijn. Kan ook weg. 1452 verschillende locaties in de kolom 'region'. Hier willen we staten van maken.
amerikaansbier.value_counts('region')

##################################################################

# ##################################################################
# 
# DATA CLEANEN. NIEUWE KOLOMMEN MAKEN 
# Nieuwe kolom 'state' uit 'region'
# De rating kolom opschonen. Strings er uit
# De ABV kolom opschonen
# De IBU kolom opschonen
# 
# ##################################################################

# Nieuwe kolom 'state' toevoegen die de staat haalt uit de kolom 'region'.

# Functie om de staat te extraheren
def extract_state(region):
    parts = region.split(", ")
    if len(parts) > 1:
        return parts[-1]  # Pak het laatste deel (de staat)
    else:
        return region  # Het is al een staat

# Nieuwe kolom toevoegen met alleen de staat
amerikaansbier["state"] = amerikaansbier["region"].apply(extract_state)

statenlijst = amerikaansbier.value_counts('state')

city_to_state = {
    "Baltimore": "Maryland",
    "Bellingham": "Washington",
    "Boulder": "Colorado",
    "Braintree": "Massachusetts",
    "Brooklyn": "New York",
    "Cambridge": "Massachusetts",
    "Canton": "Ohio",
    "Cape May": "New Jersey",
    "Centennial": "Colorado",
    "Charlotte": "North Carolina",
    "Chester": "Pennsylvania",
    "Cincinnati": "Ohio",
    "Colorado Springs": "Colorado",
    "Columbus": "Ohio",
    "D.C.": "District of Columbia",
    "Des Moines": "Iowa",
    "Detroit": "Michigan",
    "Devon": "Pennsylvania",
    "Denver": "Colorado",
    "Duluth": "Minnesota",
    "Everett": "Washington",
    "Framingham": "Massachusetts",
    "Goliad": "Texas",
    "Hampshire County": "Massachusetts",
    "Hartford": "Connecticut",
    "Honolulu": "Hawaii",
    "Houston": "Texas",
    "Iowa City": "Iowa",
    "Itasca": "Illinois",
    "Jersey City": "New Jersey",
    "Las Vegas": "Nevada",
    "Lexington": "Kentucky",
    "Lincolnshire": "Illinois",
    "Los Angeles": "California",
    "Louisville": "Kentucky",
    "Maryland Heights": "Missouri",
    "Memphis": "Tennessee",
    "Miami": "Florida",
    "Michigan City": "Indiana",
    "Milwaukee": "Wisconsin",
    "Monmouth": "Illinois",
    "Nashville": "Tennessee",
    "New Avalon": "New York",
    "Oakland": "California",
    "Phoenix": "Arizona",
    "Pittsburgh": "Pennsylvania",
    "Plymouth": "Massachusetts",
    "Portland": "Oregon",
    "Rutland": "Vermont",
    "Sacramento": "California",
    "San Francisco": "California",
    "Saint Paul": "Minnesota",
    "Santa Barbara": "California",
    "San Diego": "California",
    'Seattle': "Washington",
    "Scottsdale": "Arizona",
    "Somerville": "Massachusetts",
    "Springfield": "Illinois",
    "St. Charles": "Missouri",
    "St. Louis": "Missouri",
    "St. Paul": "Minnesota",
    "Tacoma": "Washington",
    "Tampa": "Florida",
    "Tulsa": "Oklahoma",
    "Villa Park": "Illinois",
    "Washington D.C.": "District of Columbia",
    "West Kill": "New York",
    "Wichita": "Kansas",
    "Bradley Brew Unicorn Girls American Pale Ale": "California",
    "New Avalon Brewing Company": "New York",
    "New York City": "New York",
    "Dallas": "Texas",
    "Salt Lake City": "Utah",
    "New Orleans": "Louisiana",
    "Virginia Beach": "Virginia",
    "Oklahoma City": "Oklahoma",
    "Kansas City":"Missouri",
    "Boston": "Massachusetts",
    "Minneapolis": "Minnesota",
    "Indianapolis": "Indiana",
    "Philadelphia": "Pennsylvania",
    "Atlanta": "Georgia",
    "Chicago": "Illinois"
}


# Functie om de juiste staat te vinden
def map_to_state(location):
    return city_to_state.get(location, location)  # Vervang stad door staat als die in de mapping staat

# Pas de functie toe
amerikaansbier["state"] = amerikaansbier["state"].apply(map_to_state)

valid_states = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
}

# Filter de data om alleen rijen te behouden met een geldige staat
amerikaansbier = amerikaansbier[amerikaansbier["state"].isin(valid_states)]

# Nu zijn er nog maar 50 staten (en DC), dus 51 unique values in 'states', over.

# ##################################################################

# Clean up de rating kolom

amerikaansbier["rating"] = amerikaansbier["rating"].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

# ##################################################################
# Clean up de IBU kolom

amerikaansbier["ibu"] = amerikaansbier["ibu"].astype(str)

# Filter alleen numerieke waarden en zet negatieve getallen om naar positieve integers
amerikaansbier = amerikaansbier[amerikaansbier["ibu"].str.match(r"^-?\d+$")]  # Houd alleen integer getallen (positief of negatief)
amerikaansbier["ibu"] = amerikaansbier["ibu"].astype(int).abs()  # Zet om naar integer en neem absolute waarde

##################################################################
# Clean up de ABV (alcoholpercentage) kolom

# Verwijder alles behalve getallen en de punt, zet om naar float
amerikaansbier["abv"] = (amerikaansbier["abv"].astype(str)                           # Zorg dat alles een string is
    .str.extract(r"(\d+\.\d+)")            # Haal alleen numerieke waarden met een punt
    .astype(float)                         # Zet om naar float
)

# Verwijder rijen waar 'abv' NaN is (die konden niet worden geconverteerd)
amerikaansbier = amerikaansbier.dropna(subset=["abv"])

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Bereid trainingsdata voor (alleen Amerikaanse bieren met bekende rating)
train_df = amerikaansbier.dropna(subset=["rating", "abv", "ibu"])

############### RANDOM FORREST

X_train = train_df[["abv", "ibu"]]
y_train = train_df["rating"]

# Model trainen
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Filter internationale bieren (land ≠ USA)
wereldbier = df[df["country"] != "United States"].copy()
wereldbier = wereldbier.dropna(subset=["abv", "ibu"])  # Zorg dat de inputs beschikbaar zijn

# Zet ABV en IBU kolommen om naar numeriek als dat nog niet is gebeurd
wereldbier["abv"] = wereldbier["abv"].astype(str).str.extract(r"(\d+\.\d+)").astype(float)
wereldbier["ibu"] = wereldbier["ibu"].astype(str)
wereldbier = wereldbier[wereldbier["ibu"].str.match(r"^-?\d+$")]
wereldbier["ibu"] = wereldbier["ibu"].astype(int).abs()

# Voorspel ratings
wereldbier["predicted_rating"] = rf.predict(wereldbier[["abv", "ibu"]])

# Gemiddelde predicted rating per land
wereld_rating = wereldbier.groupby("country")["predicted_rating"].mean().reset_index()

# Laad wereldkaart GeoJSON
wereldkaart_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
wereld_geojson = requests.get(wereldkaart_url).json()

# Data klaarmaken: kolomnamen moeten kloppen
wereld_rating.columns = ["country", "avg_predicted_rating"]

############### Tot hier is nieuw random forrest

# Resultaat tonen
# amerikaansbier["abv"].value_counts().to_csv('1234.csv')


##################################################################
# 
# PLOTJES: WORDEN NIET GEBRUIKT IN DE APP...
# 
##################################################################


# %%
# Plotje maken met de rating per staat
# De NaNs moeten gedropt.
# usarating = amerikaansbier.dropna(subset=["rating"])

# rating_staat = usarating.groupby('state')['rating'].mean().sort_values(ascending=False).index

# usarating_sort_rating = usarating.sort_values('rating')

# sns.barplot(data=usarating_sort_rating,x='state',y='rating',order=rating_staat)
# plt.xticks(rotation=90)

# plt.show()

# %%
# Plotje die de gemiddelde IBU waarde per staat laat zien.

# iburating = amerikaansbier.dropna(subset = ['ibu'])

# ibu_staat = iburating.groupby('state')['ibu'].mean().sort_values(ascending=False).index

# usaibu_sort_ibu = iburating.sort_values('ibu')

# sns.barplot(data=usaibu_sort_ibu,x='state',y='ibu',order=ibu_staat)
# plt.xticks(rotation=90)

# plt.show()

##################################################################
# 
# 3 KAARTEN MAKEN: WORDEN WEL GEBRUIKT IN DE APP
# HIERONDER WORDEN EERST DE JUISTE VARIABELEN GEMAAKT OM 
# DE KAARTEN TE KUNNEN MAKEN:
#
#  
##################################################################

# %%

# %%
# URL naar je GeoJSON-bestand in je GitHub repo
url = "https://raw.githubusercontent.com/Yuri194870/usbeerapp/main/us-states.json"

# Download en sla lokaal op
geojson_path = "us-states.json"
response = requests.get(url)

if response.status_code == 200:
    with open(geojson_path, "wb") as f:
        f.write(response.content)
    print("✅ GeoJSON succesvol gedownload!")

    # Lees het bestand in met geopandas
    statenkaart = gpd.read_file(geojson_path)
    print(statenkaart.head())  # Debug: Bekijk of de data correct is geladen
else:
    print("❌ Fout bij downloaden van GeoJSON:", response.status_code)

# statenkaart = gpd.read_file('us-states.json')
print(statenkaart.head())

# Maak een variabele met de gemiddelde rating per staat
ratingstaat = amerikaansbier.groupby('state')['rating'].agg('mean').round(1)

# Maak een variabele met de gemiddelde IBU per staat
IBUstaat = amerikaansbier.groupby('state')['ibu'].agg('mean').round(0)

# Maak een variabele met de hoeveelheid biertjes met een rating per staat
aantalbiermetrating = amerikaansbier.groupby('state')['rating'].count()

# Maak een variabele met de hoeveelheid biertjes met een IBUscore per staat
aantalbiermetIBU= amerikaansbier.groupby('state')['ibu'].count()

# Maak een variabele met het aantal biertjes per staat in de lijst.
aantalbierperstaat = amerikaansbier.groupby('state').size().reset_index(name='aantal')

# Je wilt de 2 nieuwe variabelen samenvoegen met polygonenkaart. Daarvoor moeten de kolommen hetzelfde heten.
# Het makkelijkst is om de naam van de kolom in de  GeoJson file te veranderen van 'name' naar 'state'
statenkaart = statenkaart.rename(columns={"name": "state"})

# Nu kun je die file samenvoegen met de aangemaakte variabelen. 
# Je krijgt dan 2 kolommen extra met in de ene de rating per staat en in de andere het aantal biertjes met een rating.
# De kolommen moet je nog weer even een logische naam geven, anders heten ze rating_x en rating_y.
ratingkaart = statenkaart.merge(ratingstaat, on='state')
ratingkaart = ratingkaart.merge(aantalbiermetrating, on='state')
ratingkaart = ratingkaart.rename(columns={"rating_x": "rating", "rating_y": 'aantal'})

IBUkaart = statenkaart.merge(IBUstaat, on='state')
IBUkaart = IBUkaart.merge(aantalbiermetIBU, on='state')
IBUkaart = IBUkaart.rename(columns={"ibu_x": "IBU", "ibu_y": 'aantal'})

aantalkaart = statenkaart.merge(aantalbierperstaat, on='state')
print(aantalkaart.columns)

# ratingkaart['rating'] = ratingkaart['rating'].fillna(0) # Sommige staten hebben geen biertjes met een rating. Ik kies er voor om die gewoon niet te tekenen. 
print(ratingkaart.columns)
# print(ratingkaart)
print(IBUkaart.columns)
# print(IBUkaart)

##################################################################
# 
# KAARTJE 1: RATINGPLOT
# 
##################################################################
#
# Center point for USA
USA = [37.0902, -95.7129]

# Create map
RatingPlot = folium.Map(location=USA, zoom_start=5)

# Add Choropleth layer met juiste kleurenschaal
folium.Choropleth(
    geo_data=ratingkaart,
    name='geometry',
    data=ratingkaart,
    columns=['state', 'rating'],
    key_on='feature.properties.state',
    fill_color='RdYlGn',           # Groen → Geel → Rood
    fill_opacity=0.7,
    line_opacity=1,
    legend_name='Average beer rating per state',
    nan_fill_color='lightgray',
    reverse=True                   # Zodat 5 = groen, 1 = rood
).add_to(RatingPlot)

# Function to style polygons (transparent fill, but visible border)
def style_function(feature):
    return {
        "fillColor": "transparent",  # Maakt de polygon transparant
        "color": "black",  # Zwarte randen zodat staten zichtbaar blijven
        "weight": 1,  # Dunne randen
        "fillOpacity": 0,  # Geen kleurvulling
    }

# Voeg interactieve polygonen toe
folium.GeoJson(
    ratingkaart,
    name="States",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["state", "rating",'aantal'],
        aliases=["State:", "Average Beer Rating:", 'No of Rated Beers:'],
        localize=True,
        sticky=False
    ),

).add_to(RatingPlot)

# Add LayerControl
folium.LayerControl().add_to(RatingPlot)

# Display the map
#display(RatingPlot)

##################################################################
# 
# KAARTJE 2: IBUPLOT
# 
##################################################################
# Create map
IBUplot = folium.Map(location=USA, zoom_start=5)

# Add Choropleth layer
folium.Choropleth(
    geo_data=IBUkaart,
    name='geometry',
    data=IBUkaart,
    columns=['state', 'IBU'],
    key_on='feature.properties.state',
    fill_color='YlOrRd',
    fill_opacity=0.5,
    line_opacity=1,
    legend_name='Average IBU score per state'
).add_to(IBUplot)

# Function to style polygons (transparent fill, but visible border)
def style_function(feature):
    return {
        "fillColor": "transparent",  # Maakt de polygon transparant
        "color": "black",  # Zwarte randen zodat staten zichtbaar blijven
        "weight": 1,  # Dunne randen
        "fillOpacity": 0,  # Geen kleurvulling
    }

# Voeg interactieve polygonen toe
folium.GeoJson(
    IBUkaart,
    name="States",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["state", "IBU",'aantal'],
        aliases=["State:", "Average IBU Score:", 'No of different beers:'],
        localize=True,
        sticky=False
    ),

).add_to(IBUplot)

# Add LayerControl
folium.LayerControl().add_to(IBUplot)

# Display the map
#display(IBUplot)

##################################################################
# 
# KAARTJE 3: NO OF BEERS KAARTJE
# 
##################################################################

# Create map
Aantalbierplot = folium.Map(location=USA, zoom_start=5)

# Add Choropleth layer
folium.Choropleth(
    geo_data=aantalkaart,
    name='geometry',
    data=aantalkaart,
    columns=['state', 'aantal'],
    key_on='feature.properties.state',
    fill_color='YlOrRd',
    fill_opacity=0.5,
    line_opacity=1,
    legend_name='Number of different beers per state'
).add_to(Aantalbierplot)

# Function to style polygons (transparent fill, but visible border)
def style_function(feature):
    return {
        "fillColor": "transparent",  # Maakt de polygon transparant
        "color": "black",  # Zwarte randen zodat staten zichtbaar blijven
        "weight": 1,  # Dunne randen
        "fillOpacity": 0,  # Geen kleurvulling
    }

# Voeg interactieve polygonen toe
folium.GeoJson(
    aantalkaart,
    name="States",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["state", 'aantal'],
        aliases=["State:", 'No of different beers:'],
        localize=True,
        sticky=False
    ),

).add_to(Aantalbierplot)

# Add LayerControl
folium.LayerControl().add_to(Aantalbierplot)

# Display the map
#display(Aantalbierplot)




##################################################################
#
# DASHBOARD BOUWEN 
# 
##################################################################

# 3 TABBLADEN 
# HET EERSTE TAB IS MEER EEN SOORT HOMEPAGE.
# HET TWEEDE TAB LAAT DE KAARTJES ZIEN.
# HET DERDE TAB LAAT DE SCATTERPLOT ZIEN.
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home of Beer", "Beer data per US State","Favorite US biertje","Internationale voorspelling","Top 10"])
with tab1:
    st.title("Welkom bij Bier Zonder Grenzen!🍺")
    st.markdown("### 🌍 Jouw paspoort naar de wereld van bier")
    st.markdown("Het **ENIGE ECHTE** bierdashboard...")

    st.markdown("""
    Verken de wereld van bier met ons interactieve dashboard, gebaseerd op een unieke dataset van meer dan **45.000 bieren wereldwijd**.  
    Ontdek trends, proefprofielen en voorspellingen – alles visueel, slim en vooral: smaakvol.
    """)

    st.markdown("---")
    st.markdown("## 🍻 Wat kun je hier ontdekken?")

    with st.expander("🦅 Bierdata per Amerikaanse staat"):
        st.success("""
        Interactieve kaarten tonen per staat de gemiddelde **rating**, **bitterheid (IBU)** en het **aantal unieke bieren**.  
        Je kunt ook filteren op aantal bieren of ratings vergelijken tussen staten.
        """)

    with st.expander("🇺🇸 Welk biertje past bij mij?"):
        st.info("""
        Filter op **alcoholpercentage (ABV)** en **bitterheid (IBU)** in een interactieve scatterplot.  
        Klik, ontdek en proef digitaal op basis van jouw voorkeur.
        """)

    with st.expander("🤖 Internationale smaakvoorspelling"):
        st.warning("""
        Met behulp van een Random Forest-model voorspellen we de verwachte bierbeoordeling per land.  
        Gebaseerd op kenmerken zoals ABV en IBU – getraind op Amerikaanse bierdata.
        """)

    with st.expander("🏆 Top 10 iconische bieren wereldwijd"):
        st.error("""
        Ontdek de wereldwijde toppers, gesorteerd op **rating**, **ABV** of **IBU**.  
        """)

    st.markdown("---")
    st.markdown("### ✨ Klaar om te proeven met je ogen? Navigeer via de tabs hierboven en begin jouw biervontuur!")



with tab2:
    st.header("State of Bier 🇺🇸 🦅")
    col1, col2 = st.columns([4,1],border=True)
    with col1:
        # Selectbox met 4 keuzes
        keuze = st.selectbox(
            "Selecteer de plot", 
            ("Lokale biertjes per staat", "IBUs per staat", "Rating per staat", "Serveertemperatuur per staat")
        )

        # ====== Temperatuurdata voorbereiden (voor keuze 4) ======
        amerikaansbier["serving_temp_c_num"] = amerikaansbier["serving_temp_c"].apply(
            lambda s: float(re.findall(r"\d+\.?\d*", str(s))[0]) if pd.notna(s) and re.findall(r"\d+\.?\d*", str(s)) else None
        )

        tempkaartdata = amerikaansbier.groupby("state")["serving_temp_c_num"].mean().reset_index().dropna()
        tempkaartdata = tempkaartdata.round(1)
        tempkaart = statenkaart.merge(tempkaartdata, on="state")

        # ====== Keuze logica voor de kaarten ======
        if keuze == "Lokale biertjes per staat":
            kaart = Aantalbierplot

        elif keuze == "IBUs per staat":
            kaart = IBUplot

        elif keuze == "Rating per staat":
            kaart = RatingPlot

        elif keuze == "Serveertemperatuur per staat":
            # Maak temperatuurkaart
            kaart = folium.Map(location=USA, zoom_start=5)

            folium.Choropleth(
                geo_data=tempkaart,
                name='geometry',
                data=tempkaart,
                columns=["state", "serving_temp_c_num"],
                key_on='feature.properties.state',
                fill_color='RdBu_r',  # Blauw = koud, rood = warm
                fill_opacity=0.7,
                line_opacity=1,
                legend_name='Gemiddelde Serveertemperatuur per Staat (°C)',
                nan_fill_color='lightgray'
            ).add_to(kaart)

            # Interactieve tooltips
            folium.GeoJson(
                tempkaart,
                name="States",
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=["state", "serving_temp_c_num"],
                    aliases=["State:", "Avg Serving Temp (°C):"],
                    localize=True,
                    sticky=False
                )
            ).add_to(kaart)

            folium.LayerControl().add_to(kaart)

        # Toon de gekozen kaart
        st_map = st_folium(kaart, width=1500, height=800)

        state_name = ''
        if st_map['last_active_drawing']:
            state_name = (st_map["last_active_drawing"]['properties']['state'])
    
    with col2:
        st.markdown(f"<b><h2>{state_name} Beer facts</h2></b>", unsafe_allow_html=True)
        st.write('''---''')
        # st.write(state_name)
        gefilterddf = amerikaansbier.copy()
        if state_name:
            gefilterddf = amerikaansbier[amerikaansbier['state'] == state_name]
        else:
            gefilterddf = gefilterddf

        toplocaties = gefilterddf["region"].value_counts().head(3)
        
        if not gefilterddf['rating'].isnull().all():  # Controleer of er ten minste één rating beschikbaar is
            gemiddelderating = gefilterddf['rating'].mean().round(1)
        else:
            # Standaardwaarde of melding wanneer er geen ratings beschikbaar zijn
            gemiddelderating = "-"  # Of gebruik 0, afhankelijk van wat je wilt weergeven

        gemiddeldeIBU = gefilterddf['ibu'].mean().astype('int')
        gemiddeldeABV = gefilterddf['abv'].mean().round(1)

       
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="No of beers", value=f"{len(gefilterddf)}")
            st.metric(label="Avg ABV", value=f"{gemiddeldeABV} %")
        with col2:
            st.metric(label="Avg Rating", value=f"{gemiddelderating}")
            st.metric(label="Avg IBUs", value=f"{gemiddeldeIBU}")
        st.write('''---''')

        # st.markdown(f"<b><h4>{toplocaties['region']} </h4></b>", unsafe_allow_html=True)
        st.markdown("<b><h5>Favoriete locaties:</h5></b>", unsafe_allow_html=True)
        for i, (region, count) in enumerate(toplocaties.items(), start=1):
            st.markdown(f"###### *{i}. {region}* - {count} beers")
        # st.markdown(f"<b><h4>{gemiddelderating} </h4></b>", unsafe_allow_html=True)
        # st.markdown(f"<b><h4>{gemiddeldeIBU} </h4></b>", unsafe_allow_html=True)
        # st.markdown(f"<b><h4>{gemiddeldeABV} %</h4></b>", unsafe_allow_html=True)

        st.write('''---''')

        st.markdown("<b><h5>Biertjes met de hoogste rating:</h5></b>", unsafe_allow_html=True)

        # Controleer of er biertjes zijn met een rating in de gefilterde DataFrame
        if 'rating' in gefilterddf.columns and not gefilterddf['rating'].isnull().all():
            # Sorteer de DataFrame op rating in aflopende volgorde en selecteer de top 3
            top_3_biers = gefilterddf.dropna(subset=['rating']).sort_values(by='rating', ascending=False).reset_index().head(3)

            # Toon de top 3 biertjes, maar alleen als er biertjes met een rating zijn
            if not top_3_biers.empty:
                for i, row in top_3_biers.iterrows():
                    st.markdown(f"###### *{i+1}. {row['name']}*")
            else:
                st.write("Er zijn geen biertjes met een rating voor deze staat.")
        else:
            st.write("Er zijn geen biertjes met een rating voor deze staat.")

       

    
    
###################### SCATTERPLOT ######################
with tab3:
    col1, col2 = st.columns([3,1], border=False)
    with col1:

        filtered_df = amerikaansbier[amerikaansbier['rating'].notnull()]
        filtered_df = filtered_df[filtered_df['food_pairing']!= '#ERROR!']

        # Unieke kleuren per staat (gesorteerd op alfabetische volgorde)
        unique_states = sorted(filtered_df["state"].unique())  
        state_color_map = {state: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, state in enumerate(unique_states)}

        # ---- 1. Maak een lege container voor de grafiek ----
        graph_placeholder = st.empty()
    st.write('''---''')

    with col2:
        st.header(" Vind jouw favorite Amerikaanse biertje! 🦅 🇺🇸")
        st.write('''---''')
        # ---- 2. Plaats de sliders onder de grafiek ----
        
        # Splits alle food pairings en haal unieke waarden op
        # Verwijder de puntjes en maak de waarde correcter
        def clean_food_name(food):
            return food.strip().replace('.', '').title()
        
        unique_food_pairings = set()
        filtered_df["food_pairing"].dropna().str.lower().str.split(", ").apply(unique_food_pairings.update)
        unique_food_pairings = sorted(food.title() for food in unique_food_pairings)
        cleaned_food_pairings = sorted(set(clean_food_name(food) for food in unique_food_pairings))


        selected_foods = st.multiselect("Kies je food paring", cleaned_food_pairings)
        st.write("")
        st.write("""---""")

        values = st.slider(
            "Selecteer IBUs", 
            int(filtered_df["ibu"].min()), 
            int(filtered_df["ibu"].max()), 
            (filtered_df["ibu"].min(), filtered_df["ibu"].max())
        )
        st.write("")
        st.write("""---""")

        values2 = st.slider(
            "Selecteer ABV %", 
            float(filtered_df["abv"].min()), 
            float(filtered_df["abv"].max()), 
            (filtered_df["abv"].min(), filtered_df["abv"].max())
        )
        st.write('''---''')

    # ---- 3. Filter de data op basis van de slider ----
    filtered_df = filtered_df[
        (filtered_df["ibu"] >= values[0]) & (filtered_df["ibu"] <= values[1]) &
        (filtered_df["abv"] >= values2[0]) & (filtered_df["abv"] <= values2[1])
    ]
    # Filter dataset: check of een geselecteerde food pairing in de kolom voorkomt
    if selected_foods:
        filtered_df = filtered_df[
            filtered_df["food_pairing"].dropna().apply(lambda x: any(food in x for food in selected_foods))
        ]
    else:
        filtered_df = filtered_df  # Geen selectie betekent geen filtering

     # ---- 4. Maak de grafiek ----
    fig_filtered = go.Figure()

    for state in unique_states:
        df_state = filtered_df[filtered_df["state"] == state]
        

        
        fig_filtered.add_trace(go.Scatter(
            x=df_state["ibu"],  
            y=df_state["abv"],
            mode="markers",
            marker=dict(
                color=state_color_map[state],
                size=(df_state["rating"]**3)/5,  # Schaal de grootte op basis van rating
                # sizemode='area',  # Zorgt voor een betere visuele schaal
                sizemin=3,  # Minimaal formaat zodat lage ratings nog zichtbaar zijn
                # sizeref=100  # Schaalverhouding aanpassen
            ),
            name=state,  
            text=df_state.apply(lambda row: f"<b>{row['name']}</b><br>Brouwer: {row['brewery']}<br><br>Rating: {row['rating']}<br>Food Pairing: {row['food_pairing']}<br>Serving Temp: {row['serving_temp_c']}°C / {row['serving_temp_f']}°F<br>IBU: {row['ibu']}<br>ABV: {row['abv']}%", axis=1),
            hoverinfo='text',
        ))
    
    

    # ---- 5. Pas de legenda aan ----
    fig_filtered.update_layout(
        # title="Alcoholpercentage VS Bitterheid",
        xaxis_title="IBU (Bitterheid)",
        yaxis_title="ABV (Alcoholpercentage)",
        template="plotly_white",
        legend_title="State",
        legend=dict(
            orientation="h",  # Horizontale legenda
            x=0,
            y=-0.2,  # Lager zetten zodat hij niet over de grafiek valt
            xanchor="left",
            yanchor="top",
            traceorder="normal",  # Behoud alfabetische volgorde
            itemsizing="constant",
        ),
        width=800,  
        height=600  
    )

    # ---- 6. Update de lege container met de grafiek ----
    graph_placeholder.plotly_chart(fig_filtered)
    col1, col2 = st.columns(2, border=True)
    with col1:
        medailles = ["🥇", "🥈", "🥉"]  # Emoji's voor de top 3
        top_3_biers = filtered_df.nlargest(3, 'rating').reset_index()  # Gebruik 'rating' als de kolomnaam voor de beoordeling
        st.markdown("<h1 style='text-align: center; color: grey;'>🏆 De beste biertjes</h1>", unsafe_allow_html=True)
        st.write('''---''')

        for i, row in top_3_biers.iterrows():
            st.markdown(f"<b><h4>{medailles[i]} {row['name']}</h4></b>", unsafe_allow_html=True)
            st.markdown(f"*{row['brewery']} - {row['state']}*")
            st.write(f"{row['sub_category_2']}")
            st.write(f"Rating: {row['rating']} - IBU: {row['ibu']} - ABV: {row['abv']}%")
            st.write(f"Food Pairing: {row['food_pairing']}")
            st.write('''---''')


    with col2:
        medailles = ["🤮", "🤢", "🤒"]  # Emoji's voor de top 3
        top_3_biers = filtered_df.nsmallest(3, 'rating').reset_index()  # Gebruik 'rating' als de kolomnaam voor de beoordeling
        st.markdown("<h1 style='text-align: center; color: grey;'>💩 De slechtste biertjes</h1>", unsafe_allow_html=True)
        st.write('''---''')

        for i, row in top_3_biers.iterrows():
            st.markdown(f"<b><h4>{medailles[i]} {row['name']}</h4></b>", unsafe_allow_html=True)
            st.markdown(f"*{row['brewery']} - {row['state']}*")
            st.write(f"{row['sub_category_2']}")
            st.write(f"Rating: {row['rating']} - IBU: {row['ibu']} - ABV: {row['abv']}%")
            st.write(f"Food Pairing: {row['food_pairing']}")
            st.write('''---''')



with tab4:
    st.header("🌍 Internationale smaakvoorspelling 🤖")
    st.markdown(
        "Gebruik de onderstaande selectie om te kiezen of je de gemiddelde **voorspelde rating** "
        "of de **gemiddelde IBU-score** per land wilt bekijken (op basis van een Random Forest model dat getraind is op Amerikaanse bieren)."
    )

    # ➕ Continent toevoegen via country_converter
    wereldbier["continent"] = coco.convert(names=wereldbier["country"], to="continent")
    wereldbier = wereldbier[wereldbier["continent"] != "not found"]

    # 🔧 Keuze tussen dropdown of slider
    filter_type = st.radio("Kies hoe je wilt filteren:", ["Dropdown-menu", "De beroemde ✨continent-slider✨"], horizontal=True)

    continenten = sorted(wereldbier["continent"].dropna().unique())
    gekozen_continent = None

    if filter_type == "Dropdown-menu":
        gekozen_continent = st.selectbox("Kies een continent", ["Alle continenten"] + continenten)
    else:
        gekozen_continent = st.select_slider("Kies een continent", options=continenten)

    # 📊 Aggregaties
    wereldbier_IBU = wereldbier.dropna(subset=["ibu"])
    ibu_per_land = wereldbier_IBU.groupby("country")["ibu"].mean().reset_index()
    ibu_per_land.columns = ["country", "avg_ibu"]

    wereld_rating = wereldbier.groupby("country")["predicted_rating"].mean().reset_index()
    wereld_rating.columns = ["country", "avg_predicted_rating"]

    # Voeg continenten toe aan kaartdata
    wereld_rating = wereld_rating.merge(wereldbier[["country", "continent"]].drop_duplicates(), on="country", how="left")
    ibu_per_land = ibu_per_land.merge(wereldbier[["country", "continent"]].drop_duplicates(), on="country", how="left")

    # 🔘 Waarde om te tonen
    kaartkeuze = st.radio("Welke waarde wil je tonen?", ["Voorspelde rating per land", "Gemiddelde IBU-score per land"])

    if kaartkeuze == "Voorspelde rating per land":
        kaartdata = wereld_rating.copy()
        waarde_kolom = "avg_predicted_rating"
        kleur = "YlGnBu"
        legenda = "Gemiddelde voorspelde bierrating"
    else:
        kaartdata = ibu_per_land.copy()
        waarde_kolom = "avg_ibu"
        kleur = "OrRd"
        legenda = "Gemiddelde IBU-score per land"

    # 🌍 Filter op continent
    if gekozen_continent and gekozen_continent != "Alle continenten":
        kaartdata = kaartdata[kaartdata["continent"] == gekozen_continent]

    # 🌐 Wereldkaart ophalen
    wereldkaart_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
    wereld_geojson = requests.get(wereldkaart_url).json()

    # 🏷️ Tooltip toevoegen
    for feature in wereld_geojson["features"]:
        land = feature["properties"]["name"]
        match = kaartdata[kaartdata["country"] == land]
        if not match.empty:
            waarde = round(match[waarde_kolom].values[0], 2)
            feature["properties"]["tooltip"] = f"{land}<br>{waarde_kolom}: {waarde}"
        else:
            feature["properties"]["tooltip"] = f"{land}<br>Geen data beschikbaar"

    # 🗺️ Folium kaart bouwen
    wereldmap = folium.Map(location=[20, 0], zoom_start=2)

    folium.Choropleth(
        geo_data=wereld_geojson,
        data=kaartdata,
        columns=["country", waarde_kolom],
        key_on="feature.properties.name",
        fill_color=kleur,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legenda,
        nan_fill_color="lightgray"
    ).add_to(wereldmap)

    folium.GeoJson(
        wereld_geojson,
        style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=["tooltip"], aliases=[""], localize=True)
    ).add_to(wereldmap)

    # 🌍 Kaart tonen
    st_folium(wereldmap, width=1400, height=700)



with tab5:
    st.header("🧪 Proefprofiel van de Planeet")
    st.markdown("Ontdek 10 iconische bieren van over de hele wereld – en zie wat onze voorspelde rating is op basis van het Amerikaanse smaakprofiel.")

    # 📄 CSV inlezen en kolomnamen normaliseren
    brew_df = pd.read_csv("Top_10_Bieren.csv")
    brew_df.columns = brew_df.columns.str.lower()

    # 🇺🇳 Landen en vlaggen
    flags = {
        "Germany": "🇩🇪",
        "Belgium": "🇧🇪",
        "Japan": "🇯🇵",
        "Mexico": "🇲🇽",
        "Australia": "🇦🇺",
        "Italy": "🇮🇹",
        "Canada": "🇨🇦",
        "United Kingdom": "🇬🇧",
        "Netherlands": "🇳🇱",
        "Brazil": "🇧🇷"
    }
    brew_df["flag"] = brew_df["country"].map(flags)

    # 🔲 Sorteeroptie als radiobuttons naast de grafiek
    grafiek_col, knoppen_col = st.columns([4, 1])
    with knoppen_col:
        st.subheader("Sorteer op:")
        sort_optie = st.radio(
            label="",
            options=["predicted_rating", "abv", "ibu"],
            index=0,
            format_func=lambda x: {
                "predicted_rating": "Rating",
                "abv": "Alcohol (%)",
                "ibu": "IBU"
            }[x]
        )

    # Sorteer de data
    sorted_df = brew_df.sort_values(by=sort_optie, ascending=False).reset_index(drop=True)

    # 📈 Plotly scatterplot
    with grafiek_col:
        fig = px.scatter(
            sorted_df,
            x="ibu",
            y="abv",
            size="predicted_rating",
            color="country",
            hover_name="name",
            hover_data=["brewery", "predicted_rating", "food_pairing"],
            labels={
                "ibu": "IBU (Bitterheid)",
                "abv": "ABV (%)",
                "predicted_rating": "Voorspelde Rating"
            },
            title=f"Top 10 Internationale Bieren – Gesorteerd op {sort_optie.capitalize()}",
            width=900,
            height=600
        )
        st.plotly_chart(fig)

    # 🏆 Podium weergeven
    st.markdown("### 🏆 Podium – Top 3 Bieren")
    podium_col1, podium_col2, podium_col3 = st.columns([1, 1.2, 1])

    with podium_col2:
        st.markdown(f"<div style='text-align:center; font-size:22px;'>🥇<br><b>{sorted_df.iloc[0]['name']}</b><br>{sorted_df.iloc[0]['flag']} *{sorted_df.iloc[0]['country']}*<br>⭐ {sort_optie.capitalize()}: {sorted_df.iloc[0][sort_optie]}</div>", unsafe_allow_html=True)

    with podium_col1:
        st.markdown(f"<div style='text-align:center; margin-top:20px;'>🥈<br><b>{sorted_df.iloc[1]['name']}</b><br>{sorted_df.iloc[1]['flag']} *{sorted_df.iloc[1]['country']}*<br>⭐ {sort_optie.capitalize()}: {sorted_df.iloc[1][sort_optie]}</div>", unsafe_allow_html=True)

    with podium_col3:
        st.markdown(f"<div style='text-align:center; margin-top:30px;'>🥉<br><b>{sorted_df.iloc[2]['name']}</b><br>{sorted_df.iloc[2]['flag']} *{sorted_df.iloc[2]['country']}*<br>⭐ {sort_optie.capitalize()}: {sorted_df.iloc[2][sort_optie]}</div>", unsafe_allow_html=True)

    # 📋 Posities 4–10
    st.markdown("---")
    st.markdown("### 📋 Plek 4 t/m 10")
    for i in range(3, 10):
        bier = sorted_df.iloc[i]
        st.markdown(f"**{i+1}. {bier['name']}** – {bier['flag']} *{bier['country']}* ({sort_optie.capitalize()}: {bier[sort_optie]})")

