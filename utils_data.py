import os
import sys
import logging
import json
import time

from datetime import datetime
from jikanpy import Jikan

logging.basicConfig(level=logging.ERROR, filename="logging_errors.txt")

class Jikan_Collector(Jikan):
    def __init__(self, data_path="data"):
        super().__init__()
        self.sleep_time = 3
        self.data_path = data_path
        
    def save_season_data(self, year, season):
        seasonal_animes = self.get_seasonal_animes(year, season)
        for anime_id, title in seasonal_animes.items():
            metadata = self.get_metadata(anime_id)
            reviews = self.get_reviews(anime_id)
            stats = self.get_stats(anime_id)
        
            try:
                metadata_path = os.path.join(self.data_path, title, "metadata.json")
                reviews_path = os.path.join(self.data_path, title, "reviews.json")
                stats_path = os.path.join(self.data_path, title, "stats.json")
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
                
                with open(reviews_path, "w") as f:
                    json.dump(reviews, f)
                    
                with open(stats_path, "w") as f:
                    json.dump(stats, f)
                    
            except Exception as e:
                logging.exception(season + str(year) + ": failed to save data for " + 
                                  str(anime_id) + " " + title)
        
    def get_seasonal_animes(self, year, season):
        seasonal_animes = {}
        try:
            seasonal_shows = self.season(year=year, season=season)['anime']
            time.sleep(self.sleep_time)
        
            for show in seasonal_shows:
                if seasonal_animes:
                    air_date = datetime.strptime(shows['airing_start'].split("+")[0], 
                                                 "%Y-%m-%dT%H:%M:%S")
                    if 2008 <= air_date.year <= 2018 and shows['type'] == 'TV':
                        seasonal_animes[show['mal_id']] = show['title']
                        
        except Exception as e:
            logging.exception(season + str(year) + ": failed to get data for " + 
                              str(anime_id) + " " + title)
                    
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
        page = 1
        while (attempts < max_attempts):
            try:
                while (page < MAX_REVIEW_PAGES):
                    review = self.anime(anime_id, extension='reviews', page=page)['reviews]
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
                stats = jikan.anime(anime_id, extension='stats')
                break
            
            except Exception as e:
                attempts += 1
                logging.exception(str(anime_id) + ": stats attempt #" + str(attempts))
            
            finally:
                time.sleep(self.time)
        
        return stats
