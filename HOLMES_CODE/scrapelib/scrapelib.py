# -*- coding: utf-8 -*-

import selenium
import sys
import urllib.request
from urllib.request import Request, urlopen
from urllib.request import URLError, HTTPError
from urllib.parse import quote
import http.client
from http.client import IncompleteRead, BadStatusLine

http.client._MAXHEADERS = 1000

import time  # Importing the time library to check the time of code execution
import os
import ssl
import datetime
import json
import re
import codecs
import socket

from bs4 import BeautifulSoup
from copy import copy
from tqdm.notebook import tqdm
from fake_useragent import UserAgent
import requests
import shutil
import re
from func_timeout import func_timeout, FunctionTimedOut
from PIL import Image

"""**Google Arguments**"""

def get_arguments(meronym, holonym, root='Image scraping', limit_g=40):

  keywords = holonym + ' ' + meronym
  output_directory = os.path.join(root, holonym)
  image_directory = meronym
  save_source = meronym
  prefix = "google_" # "word to be prefixed in front of each image name"
  limit = limit_g # limit of images to be downloaded
  chromedriver = '/usr/lib/chromium-browser/chromedriver' # specify the path to chromedriver executable in your local machine
  
  # similar images search set to false by default
  similar_images = False # 'downloads images very similar to the image URL you provide'
  
  no_numbering = False # Allows you to exclude the default numbering of images
  print_urls = False # Print the URLs of the images
  safe_search = True # Turns on the safe search filter while searching for images

  arguments = {
      'keywords' : keywords,
      'limit' : limit, 
      'output_directory' : output_directory,
      'image_directory' : image_directory,
      'similar_images' : similar_images,
      'print_urls' : print_urls,
      'prefix' : prefix,
      'chromedriver' : chromedriver,
      'safe_search' : safe_search,
      'no_numbering' : no_numbering,
      'save_source' : save_source,
  }

  return arguments

args_list = ["keywords", "limit", "output_directory", "image_directory", "similar_images",
             "print_urls", "prefix", "chromedriver", "safe_search", "no_numbering", "save_source"]
             
def create_directories(main_directory, dir_name):
        # make a search keyword  directory
        try:
            if not os.path.exists(main_directory):
                os.makedirs(main_directory)
                time.sleep(0.15)
                path = (dir_name)
                sub_directory = os.path.join(main_directory, path)
                if not os.path.exists(sub_directory):
                    os.makedirs(sub_directory)
            else:
                path = (dir_name)
                sub_directory = os.path.join(main_directory, path)
                if not os.path.exists(sub_directory):
                    os.makedirs(sub_directory)
        except OSError as e:
            if e.errno != 17:
                raise
            pass
        return

"""**Google**"""

class googleimagesdownload:
    def __init__(self):
        pass

    def _extract_data_pack(self, page):
        start_line = page.find("AF_initDataCallback({key: \\'ds:1\\'") - 10
        start_object = page.find('[', start_line + 1)
        end_object = page.rfind(']',0,page.find('</script>', start_object + 1))+1
        object_raw = str(page[start_object:end_object])
        return bytes(object_raw, "utf-8").decode("unicode_escape")

    def _image_objects_from_pack(self, data):
        image_objects = json.loads(data)[31][0][12][2]
        image_objects = [x for x in image_objects if x[0] == 1]
        return image_objects

    # Downloading entire Web Document (Raw Page Content)
    def download_page(self, url):
        print(url)
        version = (3, 0)
        cur_version = sys.version_info
        headers = {}
        headers[
            'User-Agent'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36"
        if cur_version >= version:  # If the Current Version of Python is 3.0 or above
            try:
                req = urllib.request.Request(url, headers=headers)
                resp = urllib.request.urlopen(req)
                respData = str(resp.read())
            except Exception as e:
                print("Could not open URL. Please check your internet connection and/or ssl settings \n"
                      "If you are using proxy, make sure your proxy settings is configured correctly")
        else:  # If the Current Version of Python is 2.x
            try:
                req = urllib2.Request(url, headers=headers)
                try:
                    response = urllib2.urlopen(req)
                except URLError:  # Handling SSL certificate failed
                    context = ssl._create_unverified_context()
                    response = urlopen(req, context=context)
                respData = response.read()
            except Exception as e:
                print("Could not open URL. Please check your internet connection and/or ssl settings \n"
                      "If you are using proxy, make sure your proxy settings is configured correctly")
                return "Page Not found"
        try:
            return self._image_objects_from_pack(self._extract_data_pack(respData)), self.get_all_tabs(respData)
        except Exception as e:
            print('Image objects data unpacking failed.')

    # Finding 'Next Image' from the given raw page
    def get_next_tab(self, s):
        start_line = s.find('class="dtviD"')
        if start_line == -1:  # If no links are found then give an error!
            end_quote = 0
            link = "no_tabs"
            return link, '', end_quote
        else:
            start_line = s.find('class="dtviD"')
            start_content = s.find('href="', start_line + 1)
            end_content = s.find('">', start_content + 1)
            url_item = "https://www.google.com" + str(s[start_content + 6:end_content])
            url_item = url_item.replace('&amp;', '&')

            start_line_2 = s.find('class="dtviD"')
            s = s.replace('&amp;', '&')
            start_content_2 = s.find(':', start_line_2 + 1)
            end_content_2 = s.find('&usg=', start_content_2 + 1)
            url_item_name = str(s[start_content_2 + 1:end_content_2])

            chars = url_item_name.find(',g_1:')
            chars_end = url_item_name.find(":", chars + 6)
            if chars_end == -1:
                updated_item_name = (url_item_name[chars + 5:]).replace("+", " ")
            else:
                updated_item_name = (url_item_name[chars + 5:chars_end]).replace("+", " ")

            return url_item, updated_item_name, end_content

    # Getting all links with the help of '_images_get_next_image'
    def get_all_tabs(self, page):
        tabs = {}
        while True:
            item, item_name, end_content = self.get_next_tab(page)
            if item == "no_tabs":
                break
            else:
                if len(item_name) > 100 or item_name == "background-color":
                    break
                else:
                    tabs[item_name] = item  # Append all the links in the list named 'Links'
                    time.sleep(0.1)  # Timer could be used to slow down the request for image downloads
                    page = page[end_content:]
        return tabs

    # Format the object in readable format
    def format_object(self, object):
        data = object[1]
        main = data[3]
        info = data[9]
        if info is None:
            info = data[11]
        formatted_object = {}
        try:
            formatted_object['image_height'] = main[2]
            formatted_object['image_width'] = main[1]
            formatted_object['image_link'] = main[0]
            formatted_object['image_format'] = main[0][-1 * (len(main[0]) - main[0].rfind(".") - 1):]
            formatted_object['image_description'] = info['2003'][3]
            formatted_object['image_host'] = info['183836587'][0]
            formatted_object['image_source'] = info['2003'][2]
            formatted_object['image_thumbnail_url'] = data[2][0]
        except Exception as e:
            print(e)
            return None
        return formatted_object


    # Building URL parameters
    def build_url_parameters(self):
        built_url = "&tbs="
        return built_url

    # building main search URL
    def build_search_url(self, search_term, params, similar_images, safe_search):
        # check safe_search
        safe_search_string = "&safe=active"
        # check the args and choose the URL
        if similar_images:
            print('Looking for similar images...')
            keywordem = self.similar_images(similar_images)
            url = 'https://www.google.com' + keywordem
        else:
            url = 'https://www.google.com/search?q=' + quote(
                search_term.encode(
                    'utf-8')) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + params + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'

        # safe search check
        if safe_search:
            url = url + safe_search_string

        return url

    # Download Images
    def download_image(self, image_url, image_format, main_directory, dir_name, count, print_urls,
                       prefix, no_numbering, save_source, img_src, similar_images):
        try:
            req = Request(image_url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
            try:
                # timeout time to download an image
                timeout = 10

                response = urlopen(req, None, timeout)
                data = response.read()
                info = response.info()
                response.close()

                qmark = image_url.rfind('?')
                if qmark == -1:
                    qmark = len(image_url)
                slash = image_url.rfind('/', 0, qmark) + 1
                image_name = str(image_url[slash:qmark]).lower()

                type = info.get_content_type()
                if type == "image/jpeg" or type == "image/jpg":
                    if not image_name.endswith(".jpg") and not image_name.endswith(".jpeg"):
                        image_name += ".jpg"
                elif type == "image/png":
                    if not image_name.endswith(".png"):
                        image_name += ".png"
                elif type == "image/webp":
                    if not image_name.endswith(".webp"):
                        image_name += ".webp"
                elif type == "image/gif":
                    if not image_name.endswith(".gif"):
                        image_name += ".gif"
                elif type == "image/bmp" or type == "image/x-windows-bmp":
                    if not image_name.endswith(".bmp"):
                        image_name += ".bmp"
                elif type == "image/x-icon" or type == "image/vnd.microsoft.icon":
                    if not image_name.endswith(".ico"):
                        image_name += ".ico"
                elif type == "image/svg+xml":
                    if not image_name.endswith(".svg"):
                        image_name += ".svg"
                else:
                    download_status = 'fail'
                    download_message = "Invalid image format '" + type + "'. Skipping..."
                    return_image_name = ''
                    absolute_path = ''
                    return download_status, download_message, return_image_name, absolute_path

                # prefix name in image
                if prefix:
                    prefix = prefix + " "
                else:
                    prefix = ''

                if no_numbering:
                    path = main_directory + "/" + dir_name + "/" + (prefix + image_name).replace(" ", "")
                else:
                    path = main_directory + "/" + dir_name + "/" + (prefix + str(count) + "_" + image_name).replace(" ", "")

                try:
                    output_file = open(path, 'wb')
                    output_file.write(data)
                    output_file.close()
                    if save_source:
                        if similar_images:
                            list_path = main_directory + "/" + save_source + "_vs.txt"
                        else:
                            list_path = main_directory + "/" + save_source + ".txt"
                        list_file = open(list_path, 'a')
                        list_file.write(image_url + '\n')
                        list_file.close()
                    absolute_path = os.path.abspath(path)
                except OSError as e:
                    download_status = 'fail'
                    download_message = "OSError on an image...trying next one..." + " Error: " + str(e)
                    return_image_name = ''
                    absolute_path = ''

                # return image name back to calling method to use it for thumbnail downloads
                download_status = 'success'
                if no_numbering:
                    download_message = "Completed Image ====> " + (prefix + image_name).replace(" ", "")
                    return_image_name = (prefix + image_name).replace(" ", "")
                else:
                    download_message = "Completed Image ====> " + (prefix + str(count) + "_" + image_name).replace(" ", "")
                    return_image_name = (prefix + str(count) + "_" + image_name).replace(" ", "")

            except UnicodeEncodeError as e:
                download_status = 'fail'
                download_message = "UnicodeEncodeError on an image...trying next one..." + " Error: " + str(e)
                return_image_name = ''
                absolute_path = ''

            except URLError as e:
                download_status = 'fail'
                download_message = "URLError on an image...trying next one..." + " Error: " + str(e)
                return_image_name = ''
                absolute_path = ''

            except BadStatusLine as e:
                download_status = 'fail'
                download_message = "BadStatusLine on an image...trying next one..." + " Error: " + str(e)
                return_image_name = ''
                absolute_path = ''

        except HTTPError as e:  # If there is any HTTPError
            download_status = 'fail'
            download_message = "HTTPError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except URLError as e:
            download_status = 'fail'
            download_message = "URLError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except ssl.CertificateError as e:
            download_status = 'fail'
            download_message = "CertificateError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except IOError as e:  # If there is any IOError
            download_status = 'fail'
            download_message = "IOError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except IncompleteRead as e:
            download_status = 'fail'
            download_message = "IncompleteReadError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        return download_status, download_message, return_image_name, absolute_path

    def _get_all_items(self, image_objects, main_directory, dir_name, limit, arguments):
        items = []
        abs_path = []
        errorCount = 0
        i = 0
        count = 1
        while count < limit + 1 and i < len(image_objects):
            if len(image_objects) == 0:
                print("no_links")
                break
            else:
                # format the item for readability
                object = self.format_object(image_objects[i])

                # download the images
                download_status, download_message, return_image_name, absolute_path = self.download_image(
                    object['image_link'], object['image_format'], main_directory, dir_name, count, arguments['print_urls'],
                    arguments['prefix'], arguments['no_numbering'], arguments['save_source'], object['image_source'], arguments['similar_images'])
                print(download_message)
                if download_status == "success":
                    count += 1
                    object['image_filename'] = return_image_name
                    items.append(object)  # Append all the links in the list named 'Links'
                    abs_path.append(absolute_path)
                else:
                    errorCount += 1

            i += 1
        if count < limit:
            print("\n\nUnfortunately all " + str(
                limit) + " could not be downloaded because some images were not downloadable. " + str(
                count - 1) + " is all we got for this search filter!")
        return items, errorCount, abs_path

    # Bulk Download
    def download(self, arguments):
        paths_agg = {}
        paths, errors = self.download_executor(arguments)
        for i in paths:
            paths_agg[i] = paths[i]
        return paths_agg, errors

    def download_executor(self, arguments):
        paths = {}
        errorCount = None
        for arg in args_list:
            if arg not in arguments:
                arguments[arg] = None
        ######Initialization and Validation of user arguments
        if arguments['keywords']:
            search_keyword = [str(item) for item in arguments['keywords'].split(',')]

        # Setting limit on number of images to be downloaded
        if arguments['limit']:
            limit = int(arguments['limit'])
        else:
            limit = 100

        if arguments['similar_images']:
            current_time = str(datetime.datetime.now()).split('.')[0]
            search_keyword = [current_time.replace(":", "_")]

        # If single_image or url argument not present then keywords is mandatory argument
        if arguments['similar_images'] is None and arguments['keywords'] is None:
            print('-------------------------------\n'
                  'Uh oh! Keywords is a required argument \n\n'
                  '\n\nexiting!\n'
                  '-------------------------------')
            sys.exit()

        # If this argument is present, set the custom output directory
        if arguments['output_directory']:
            main_directory = arguments['output_directory']
        else:
            main_directory = "downloads"

        ######Initialization Complete
        total_errors = 0
        i = 0
        while i < len(search_keyword):  # for every main keyword
            iteration = "\n" + "Item no.: " + str(i + 1) + " -->" + " Item name = " + (
            search_keyword[i])
            print("Downloading images for: " + (search_keyword[i]) + " ...")
            search_term = search_keyword[i]

            if arguments['image_directory']:
                dir_name = arguments['image_directory']
            else:
                dir_name = search_term

            create_directories(main_directory, dir_name)  # create directories in OS

            params = self.build_url_parameters()

            url = self.build_search_url(search_term, params, arguments['similar_images'],
                                        arguments['safe_search'])  # building main search url

            images, tabs = self.download_page(url)  # download page

            items, errorCount, abs_path = self._get_all_items(images, main_directory, dir_name, limit,
                                                              arguments)  # get all image items and download images
            paths[search_keyword[i]] = abs_path

            i += 1
            total_errors = total_errors + errorCount
            print("\nErrors: " + str(errorCount) + "\n")
        return paths, total_errors

    def similar_images(self, similar_images):
        version = (3, 0)
        cur_version = sys.version_info
        headers = {}
        headers[
            'User-Agent'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36"
        if cur_version >= version:  # If the Current Version of Python is 3.0 or above
            try:
                searchUrl = 'https://www.google.com/searchbyimage?site=search&sa=X&image_url=' + similar_images

                req1 = urllib.request.Request(searchUrl, headers=headers)
                resp1 = urllib.request.urlopen(req1)
                content = str(resp1.read())
                l1 = content.find('AMhZZ')
                l2 = content.find('&', l1)
                urll = content[l1:l2]

                newurl = "https://www.google.com/search?tbs=sbi:" + urll + "&site=search&sa=X"
                req2 = urllib.request.Request(newurl, headers=headers)
                resp2 = urllib.request.urlopen(req2)

                soup = BeautifulSoup(resp2, features="lxml")
                link = soup.findAll('a', attrs={'class': 'ekf0x hSQtef'})[0]
                return link.get('href') 

            except Exception as e:
                return " - Could not connect to Google Images endpoint - "
        else:  # If the Current Version of Python is 2.x
            try:
                searchUrl = 'https://www.google.com/searchbyimage?site=search&sa=X&image_url=' + similar_images

                req1 = urllib2.Request(searchUrl, headers=headers)
                resp1 = urllib2.urlopen(req1)
                content = str(resp1.read())
                l1 = content.find('AMhZZ')
                l2 = content.find('&', l1)
                urll = content[l1:l2]

                newurl = "https://www.google.com/search?tbs=sbi:" + urll + "&site=search&sa=X"
                req2 = urllib2.Request(newurl, headers=headers)
                resp2 = urllib2.urlopen(req2)

                soup = BeautifulSoup(resp2)
                link = soup.findAll('a', attrs={'class': 'ekf0x hSQtef'})[0]
                return link.get('href')
            except:
                return " - Could not connect to Google Images endpoint - "

"""**Bing**"""

def error(link, query):
    print("[!] Skipping {}. Can't download or no metadata.\n".format(link))

def save_image(link, file_path):
    # Use a random user agent header for bot id
    ua = UserAgent(verify_ssl=False)
    headers = {"User-Agent": ua.random}
    r = requests.get(link, stream=True, headers=headers)
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
        # check the file is a valid image
        try:
          im = Image.open(file_path)
          im.verify() 
          im.close()
        except Exception as e: 
          if os.path.exists(file_path) == True:
            os.remove(file_path)
          raise Exception("Image not valid or corrupted.")
    else:
        raise Exception("Image returned a {} error.".format(r.status_code))

def download_image(link, image_data, metadata, query, prefix, output_directory, image_directory, save_source):
    download_image.delta += 1

    # Get the image link
    try:
        # Get the file name and type
        file_name = link.split("/")[-1]
        type = file_name.split(".")[-1]
        type = (type[:3]) if len(type) > 3 else type
        if type.lower() == "jpe":
            type = "jpeg"
        if type.lower() not in ["jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
            type = "jpg"

        # Download the image
        print("[%] Downloading Image #{} from {}".format(
            download_image.delta, link))
        try:
            if prefix is not None:
              path = "{}/{}/".format(output_directory, image_directory) + "{}_{}_{}".format(prefix, download_image.delta, file_name)
            else:
              path = "{}/{}/".format(output_directory, image_directory) + "{}_{}".format(download_image.delta, file_name)
            # assume directories already exist
            create_directories(output_directory, image_directory)

            try:
              func_timeout(10, save_image, args=(link, path))
            except FunctionTimedOut as e:
              raise Exception('Timeout')

            print("[%] Downloaded File")
            if metadata:
                list_path = output_directory + "/" + save_source + ".txt"
                list_file = open(list_path, 'a')
                list_file.write(link + '\n')
                list_file.close()
        except Exception as e:
            download_image.delta -= 1
            print("[!] Issue Downloading: {}\n[!] Error: {}".format(link, e))
            error(link, query)
    except Exception as e:
        download_image.delta -= 1
        print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))
        error(link, query)


def bing(query, delta, output_directory, image_directory, save_source, metadata=True, adult=True, prefix=None):

    delta = int(delta)

    # set stack limit
    sys.setrecursionlimit(1000000)

    page_counter = 0
    link_counter = 0
    download_image.delta = 0
    while download_image.delta < delta:
        # Parse the page source and download pics
        ua = UserAgent(verify_ssl=False)
        headers = {"User-Agent": ua.random}
        payload = (("q", str(query)), ("first", page_counter), ("adlt", adult))
        source = requests.get(
            "https://www.bing.com/images/async", params=payload, headers=headers).content
       
        soup = BeautifulSoup(str(source).replace('\r\n', ""), "lxml")

        try:
            os.remove("dataset/logs/bing/errors.log")
        except OSError:
            pass

        # Get the links and image data
        try:
          links = [json.loads(i.get("m").replace('\\', ""))["murl"]
                  for i in soup.find_all("a", class_="iusc")]
        except Exception as e:
          regex = re.compile(r'\\(?![/u"])')
          links = [json.loads(regex.sub(r"\\\\", i.get("m")))["murl"]
                  for i in soup.find_all("a", class_="iusc")]

        print("[%] Indexed {} Images on Page {}.".format(
            len(links), page_counter + 1))
        print("\n===============================================\n")
        print("[%] Getting Image Information.")
        images = {}
        for a in soup.find_all("a", class_="iusc"):
            if download_image.delta >= delta:
                break
            print("\n------------------------------------------")
            try:
              iusc = json.loads(a.get("m").replace("\\",""))
            except Exception as e:
              regex = re.compile(r'\\(?![/u"])')
              iusc = json.loads(regex.sub(r"\\\\", a.get("m")))
            link = iusc["murl"]
            print("\n[%] Getting info on: {}".format(link))
            try:
                image_data = "bing", query, link, iusc["purl"], iusc["md5"]
                images[link] = image_data
                try:
                    download_image(link, images[link], metadata, query, prefix, output_directory, image_directory, save_source)
                except Exception as e:
                    print(e)
                    error(link, query)
            except Exception as e:
                images[link] = image_data
                print("[!] Issue getting data: {}\n[!] Error: {}".format(image_data, e))

            link_counter += 1

        page_counter += 1

    print("\n\n[%] Done. Downloaded {} images.".format(download_image.delta))
    print("\n===============================================\n")

def google_image_scraping(arguments):
  records = []
  records.append(arguments)

  total_errors = 0
  t0 = time.time()  # start the timer
  for arguments in records:
    # download multiple images based on keywords/keyphrase search
    response = googleimagesdownload()
    paths, errors = response.download(arguments)  # wrapping response in a variable just for consistency
    total_errors = total_errors + errors

    t1 = time.time()  # stop the timer
    total_time = t1 - t0  # Calculating the total time required to crawl, find and download all the links of 60,000 images
    print("\nEverything downloaded!")
    print("Total errors: " + str(total_errors))
    print("Total time taken: " + str(total_time) + " Seconds")

"""**Download visually similar images**"""

#@title
def similar_images_scraping(arguments, limit=5):
    filepath = arguments['output_directory'] + "/" + arguments['save_source'] + ".txt"
    arguments_s = copy(arguments)
    arguments_s['limit'] = limit
    arguments_s['prefix'] = "related_"
    #arguments_s['save_source'] = False
    total_errors = 0
    with open(filepath) as fp:
      for url in tqdm(fp):
          print(url)
          arguments_s['similar_images'] = url
          try:
            google_image_scraping(arguments_s)
          except Exception as e:
            print(e)
            print('URLError on an image...trying next one...')
            total_errors += 1
            print('\nErrors on similar images: ' + str(total_errors) +'\n')
    print('\nTotal errors on similar images: ' + str(total_errors) +'\n')

"""**Download everything**"""

def do_scraping(meronym, arguments, limit_b):
  # google
  google_image_scraping(arguments)
  # bing
  bing(arguments['keywords'], limit_b, arguments['output_directory'], meronym, meronym, prefix="bing")

def scrape_parts(holonym, meronyms, root='Image_scraping', limit_g=40, limit_b=60):
  arguments_list = []
  # create holonym dir if not existing
  if os.path.exists(os.path.join(root, holonym)) == False:
    os.mkdir(os.path.join(root, holonym))
  for meronym in meronyms:
    arguments = get_arguments(meronym, holonym, root=root, limit_g=limit_g)
    arguments_list.append(arguments)
    do_scraping(meronym, arguments, limit_b)

  return arguments_list
