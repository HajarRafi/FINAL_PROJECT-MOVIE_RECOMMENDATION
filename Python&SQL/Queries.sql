create database movie_project;
use movie_project;

-- check the number of some values

select count(distinct(id)) 
from movies;

select count(distinct(country)) 
from production_countries;

select count(distinct(language)) 
from languages;

select count(distinct(genre))
from genres;

-- transform some tables for Tableau visualization (too much data..leave only popular ones)

sel