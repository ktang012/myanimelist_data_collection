import os
import sys
import logging
import json
import time

from datetime import datetime
from jikanpy import Jikan
from pprint import pprint

logging.basicConfig(level=logging.ERROR, filename="logging_errors.txt")

class Jikan_Collector(Jikan):
    def __init__(self, data_path="data"):
        super().__init__()
        self.sleep_time = 3
        self.data_path = data_path
        
    def save_season_data(self, year, season):
        seasonal_animes = self.get_seasonal_animes(year, season)
        for anime_id, title in seasonal_animes.items():
            print("collecting data for", title)
            self.save_anime_data(anime_id, season=season, year=year, title=title)
            print("-"*10)
                                  
    def save_anime_data(self, anime_id, max_review_pages=10, max_attempts=10,
                        season="NA", year="NA", title="NA"):
        anime_path = os.path.join(self.data_path, title + "_" + str(anime_id))
        
        if not os.path.exists(anime_path):
            os.mkdir(anime_path)
        
        try:
            metadata = self.get_metadata(anime_id, max_attempts=max_attempts)
            metadata_path = os.path.join(anime_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, sort_keys=True, indent=4, separators=(',', ': '))
                
        except Exception as e:
            logging.exception(season + str(year) + ": failed to save data for " + 
                              str(anime_id) + " " + title)
        
        try:
            reviews = self.get_reviews(anime_id, max_attempts=max_attempts,
                                       max_review_pages=max_review_pages)
            reviews_path = os.path.join(anime_path, "reviews.json")
            with open(reviews_path, "w") as f:
                json.dump(reviews, f, sort_keys=True, indent=4, separators=(',', ': '))
                
        except Exception as e:
            logging.exception(season + str(year) + ": failed to save data for " + 
                              str(anime_id) + " " + title)
        
        try:
            stats = self.get_stats(anime_id, max_attempts=max_attempts)
            stats_path = os.path.join(anime_path, "stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, sort_keys=True, indent=4, separators=(',', ': '))
                
        except Exception as e:
            logging.exception(season + str(year) + ": failed to save data for " + 
                              str(anime_id) + " " + title)
    
    def get_seasonal_animes(self, year, season, min_year=2008, max_year=2018, min_members=70000, max_attempts=5):
        seasonal_animes = {}
        attempts = 0
        while (attempts < max_attempts):
            try:
                seasonal_shows = self.season(year=year, season=season)['anime']
                break
            except:
                attempts += 1
            time.sleep(self.sleep_time)
        
        if attempts == max_attempts:
           return seasonal_animes
        
        for show in seasonal_shows:
            try:
                air_date = datetime.strptime(show['airing_start'].split("+")[0], 
                                             "%Y-%m-%dT%H:%M:%S")
                if min_year <= air_date.year <= max_year and show['type'] == 'TV' and not show['continuing'] and show['members'] >= min_members:
                    seasonal_animes[show['mal_id']] = show['title']
                        
            except Exception as e:
                continue
                    
        return seasonal_animes
           
    def get_metadata(self, anime_id, max_attempts=5):
        metadata = None
        print(" " * 5, "getting metadata for", str(anime_id))
        attempts = 0
        while (attempts < max_attempts):
            try:
                metadata = self.anime(anime_id)
                break
                
            except Exception as e:
                attempts += 1
                logging.exception(str(anime_id) + ": metadata attempt #" + str(attempts))
                
            finally:
                time.sleep(self.sleep_time)
                
        return metadata
   
    def get_reviews(self, anime_id, max_attempts=5, max_review_pages=10):
        reviews = []
        print(" " * 5, "getting reviews for", str(anime_id))
        attempts = 0
        page = 0
        while (attempts < max_attempts):
            try:
                while (page < max_review_pages):
                    review = self.anime(anime_id, extension='reviews', page=page)['reviews']
                    if review:
                        reviews.append(review)
                        page += 1
                    else:
                        break
                    time.sleep(self.sleep_time)
                break
                
            except Exception as e:
                attempts += 1
                logging.exception(str(anime_id) + ": reviews attempt #" + str(attempts))
                time.sleep(self.sleep_time)
        
        return reviews
                
    def get_stats(self, anime_id, max_attempts=5):
        stats = None
        print(" " * 5, "getting stats for", str(anime_id))
        attempts = 0
        while (attempts < max_attempts):
            try:
                stats = self.anime(anime_id, extension='stats')
                break
            
            except Exception as e:
                attempts += 1
                logging.exception(str(anime_id) + ": stats attempt #" + str(attempts))
            
            finally:
                time.sleep(self.sleep_time)
        
        return stats
