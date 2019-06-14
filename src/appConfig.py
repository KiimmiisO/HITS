#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:45:56 2018

@author: nantanasriphromting
"""
import time
from datetime import datetime

class dbConfig:
    hostname = 'localhost'
    username = 'postgres'
    password = '@dm1n'
    database = 'dbtwitter'
    
class tokenConfig:
    consumer_key = "ofkAfCP0Rsc53lm7wHvUdIO2A"
    consumer_secret = "EQAVtoBN019xWVQAp2kkpVEuWKgNFtSlztAkxAHcvv6oInIGhI"
    access_token_key = "1544158663-Co0oETTDX2WztUg3F1x27JAkS0G4xhIkcgN2sfn"
    access_token_secret = "vr3Fh99bV8BEuaXPSXqFJiIpsWYdtUdDPpf2TZYeoMTFj"   
 
def set_date(date_str):
        """Convert string to datetime
        """
        time_struct = time.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")#Tue Apr 26 08:57:55 +0000 2011
        date = datetime.fromtimestamp(time.mktime(time_struct))
        return date
print('set variable')     