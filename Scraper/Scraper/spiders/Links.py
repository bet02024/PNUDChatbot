import scrapy
from scrapy.http import Request

class LinksSpider(scrapy.Spider):
    name = 'Links'
    ##allowed_domains = ['cg-hubdev-cggis.opendata.arcgis.com']
    start_urls = ['https://cg-hubdev-cggis.opendata.arcgis.com/']
    links=[]

    def start_requests(self):
        request=Request(self.start_urls[0], cookies={'store_language':'en'}, callback=self.parse_page)
        yield request


    def parse_page(self,response):
        item=scrapy.Item()
        #content=response.xpath('//div[@id="items"]//div[@class="article-meta"]')# Loops through the each and every article link in HTML 'content'
        for article_link in response.xpath('.//a'):# Extracts the href info of the link to store in scrapy item
            print(article_link)
            yield(item)


    def parse(self, response):
        print("### PARSE  ###")

        print(response)
        self.links.append(response.url)
        for href in response.css('a::attr(href)'):
            print("#### LINKS ", href)
            yield response.follow(href, self.parse)

    def spider_closed(self):
        print(self.links)
