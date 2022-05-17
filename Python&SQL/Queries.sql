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

select movie_id, id as company_id, name as company
from companies 
where name in (select name from 
(select name, count(*) as n_movies
from companies
group by name
having n_movies > 20) sub);

select movie_id, id as company_id, name as company
from companies 
where name in (select name from 
(select name, count(*) as n_movies
from companies
group by name
having n_movies > 20) sub);


select movie_id, name as director
from directors 
where name in (select name from 
(select name, count(*) as n_movies
from directors
group by name
having n_movies > 20) sub);
