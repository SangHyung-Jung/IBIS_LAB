'''
이건 걍 크롤러 동성이가 준거 내가 좀 수정한거임
딱히 뭐.. 코드가 약간 더럽긴한데 동성이가 준게
걍 내비둬도 될듯?

교정: 이거 올려야할까? 크롤링 수준만 들키고 따지고보면 불법적인 일이라?
'''


import os
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.parse import quote
import time, random, re, math
import pandas as pd
import glob
import calendar
import csv


# 특정 경로 내 엑셀 파일 불러오기 및 통합 함수
def load_news_url(file_path):
    file_path = glob.glob(file_path + '*.xlsx')
    df = pd.DataFrame()
    for file_list in file_path:
        news_excel_file = pd.read_excel(file_list)
        df = pd.concat([df, news_excel_file])
    return df


# 특정 기사의 내용, 구독자 반응 수집 함수
def content_crawler(page_url):
    # 구글 가상 브라우징 옵션 헤더 정보
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('disable-gpu')
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36")
    options.add_experimental_option("prefs", {'profile.managed_default_content_settings.javascript': 2})

    # 구글 드라이버 경로 설정 및 옵션 적용
    driver = webdriver.Chrome('/home/ir1067/IR/news_data_title/chromedriver', chrome_options=options)
    driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: function() {return[1, 2, 3, 4, 5];},});")
    driver.execute_script("Object.defineProperty(navigator, 'languages', {get: function() {return ['ko-KR', 'ko']}})")

    # 구글 드라이버 가상 브라우징 및 페이지 스크랩
    driver.get(page_url)
    driver.implicitly_wait(2.5)
    time.sleep(random.uniform(1, 2))

    news_pages = driver.page_source.encode('utf-8')

    driver.quit()

    # 웹페이지 파싱
    news_pages = BeautifulSoup(news_pages, 'html.parser')

    # 기사 제목, 날짜, 본문 수집
    title = news_pages.find('h3', {'id': 'articleTitle'}).get_text()
    date = news_pages.find('span', {'class': 't11'}).get_text()
    raw_contents = news_pages.find('div', {'id': 'articleBodyContents'})
    contents = news_pages.find('div', {'id': 'articleBodyContents'}).get_text().strip()

    # reaction 수집
    good = re.sub("<.*?>", '', str(news_pages.find('div', {'class': '_reactionModule u_likeit'}).find_all('span', {
        'class': 'u_likeit_list_count'})[0]))
    warm = re.sub("<.*?>", '', str(news_pages.find('div', {'class': '_reactionModule u_likeit'}).find_all('span', {
        'class': 'u_likeit_list_count'})[1]))
    sad = re.sub("<.*?>", '', str(news_pages.find('div', {'class': '_reactionModule u_likeit'}).find_all('span', {
        'class': 'u_likeit_list_count'})[2]))
    angry = re.sub("<.*?>", '', str(news_pages.find('div', {'class': '_reactionModule u_likeit'}).find_all('span', {
        'class': 'u_likeit_list_count'})[3]))
    want = re.sub("<.*?>", '', str(news_pages.find('div', {'class': '_reactionModule u_likeit'}).find_all('span', {
        'class': 'u_likeit_list_count'})[4]))

    return date, title, contents, raw_contents, good, warm, sad, angry, want

if __name__ == "__main__":
    load_path = '/home/ir1067/FOR_CONTENT/'
    save_path = '/home/ir1067/FOR_CONTENT/'

    for company in ['유한양행_cwl.xlsx']:
        pd_url = pd.read_excel(load_path + "{}".format(company), index_col=0)
        # 기사 링크 값 추출
        url_lists = pd_url['link'].values

        print("=" * 30)
        print("{} crawling start!".format(company))

        date = []
        title = []
        contents = []
        raw_contents = []
        good = []
        warm = []
        sad = []
        angry = []
        want = []

        for i, url in enumerate(url_lists):
            try:
                result = content_crawler(url)

                date.append(result[0])
                title.append(result[1])
                contents.append(result[2])
                raw_contents.append(result[3])
                good.append(result[4])
                warm.append(result[5])
                sad.append(result[6])
                angry.append(result[7])
                want.append(result[8])

                del result

                print("{}/{} 수집 완료".format(i + 1, len(url_lists)))

                time.sleep(random.uniform(1, 2))

            except Exception as ex:
                print("{}".format(ex))

            if (i+1) % 3000 == 0:

                writer = pd.ExcelWriter(save_path + '{}_{}'.format(company[:-9], i+1) + '_contents.xlsx',
                                        engine='xlsxwriter',
                                        options={'strings_to_urls': False})
                contents_output = pd.DataFrame([date, title, contents, raw_contents, good, warm, sad, angry, want]).T
                contents_output.columns = ['date', 'title', 'contents', 'raw_contents', 'good', 'warm', 'sad', 'angry', 'want']

                contents_output.to_excel(writer)
                writer.close()

                del writer, contents_output
                del date, title, contents, raw_contents, good, warm, sad, angry, want

                date = []
                title = []
                contents = []
                raw_contents = []
                good = []
                warm = []
                sad = []
                angry = []
                want = []

            if (i + 1) == len(url_lists):
                writer = pd.ExcelWriter(save_path + '{}_{}'.format(company[:-9], i+1) + '_contents.xlsx',
                                        engine='xlsxwriter',
                                        options={'strings_to_urls': False})
                contents_output = pd.DataFrame([date, title, contents, raw_contents, good, warm, sad, angry, want]).T
                contents_output.columns = ['date', 'title', 'contents', 'raw_contents', 'good', 'warm', 'sad', 'angry',
                                           'want']

                contents_output.to_excel(writer)
                writer.close()

                del writer, contents_output
                del date, title, contents, raw_contents, good, warm, sad, angry, want
