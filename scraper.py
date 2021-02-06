import requests
import pandas as pd
from bs4 import BeautifulSoup

if  __name__ == '__main__':
    all_quotes = []
    url_list = []
    urls = ['https://www.wishesquotes.com/valentines-day/valentines-day-messages',
            'https://www.wishesquotes.com/valentines-day/valentines-day-messages-for-her-girlfriend-wife',
            'https://www.wishesquotes.com/valentines-day/love-sms-messages',
            'https://www.wishesquotes.com/love/love-quotes-for-her',
            'https://www.wishesquotes.com/valentines-day/valentines-day-greetings',
            'https://www.wishesalbum.com/valentines-day-messages/',
            'https://www.wishesalbum.com/valentines-day-messages-for-him/',
            'https://www.wishesalbum.com/valentines-day-wishes/',
            'https://www.wishesalbum.com/love-messages-for-romantic-hearts/',
            'https://www.wishesalbum.com/50-love-sms-for-boyfriend/',
            'https://www.wishesalbum.com/50-love-sms-for-girlfriend/',
            'https://www.wishesalbum.com/short-love-text-messages/',
            'https://www.wishesalbum.com/cute-love-quotes/',
            'https://www.wishesalbum.com/romantic-valentines-day-love-quotes/',
            'https://www.wishesalbum.com/meaningful-love-quotes/',
            'https://www.lovewishesquotes.com/50-romantic-quotes-you-should-share-with-your-love',
            'https://www.lovewishesquotes.com/30-sweet-love-quotes-young-lovers',
            'https://www.lovewishesquotes.com/40-cute-love-quotes-for-her',
            'https://www.lovewishesquotes.com/valentines-day-messages-for-him',
            'https://www.lovewishesquotes.com/valentines-day-love-messages',
            'https://www.lovewishesquotes.com/valentines-day-love-quotes',
            'https://www.lovewishesquotes.com/valentines-day-messages-for-her',
            'https://www.lovewishesquotes.com/40-unique-love-quotes-for-him',
            'https://www.lovewishesquotes.com/goodnight-messages-for-him',
            'https://www.lovewishesquotes.com/love-sayings',
            'https://www.lovewishesquotes.com/sweet-good-morning-messages-for-him',
            'https://www.lovewishesquotes.com/best-compliments-for-girls-100-ways-to-make-a-woman-feel-special',
            'https://www.lovewishesquotes.com/flirty-messages-pick-up-lines-quotes-text-your-crush']
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        article = soup.find('article')
        quotes = article.findChildren("li")
        def  filter_items(quote):
            for child in quote.children:
                if child.name == 'a' or child.name=='strong':
                    return False
            return True
        for quote in quotes:
            if filter_items(quote):
                all_quotes.append(quote.text)
                url_list.append(url)

    print(len(all_quotes))
    df = pd.DataFrame({'message': all_quotes, 'url': url_list})
    df.to_csv('valentines_messages_2.csv')
