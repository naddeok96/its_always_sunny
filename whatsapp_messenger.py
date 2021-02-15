from selenium import webdriver 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.by import By 
import time 

# Replace below path with the absolute path 
# to chromedriver in your computer 
options = webdriver.ChromeOptions()
options.add_argument('--headless') 
# options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-dev-shm-usage")
options.add_argument('--no-sandbox')
options.add_argument("--disable-gpu")
options.binary_location = "/mnt/c/Program Files (x86)/Google/Chrome/Application/"
chrome_driver_binary = '/mnt/c/Program Files (x86)/Google/Chrome/chromedriver'
driver = webdriver.Chrome(chrome_driver_binary, chrome_options=options) 

driver.get("https://web.whatsapp.com/") 
wait = WebDriverWait(driver, 600) 

# Replace 'Friend's Name' with the name of your friend 
# or the name of a group 
target = '"Kyle Naddeo"'

# Replace the below string with your own message 
string = "Message sent using Python!!!"

x_arg = '//span[contains(@title,' + target + ')]'
group_title = wait.until(EC.presence_of_element_located(( 
	By.XPATH, x_arg))) 
group_title.click() 
inp_xpath = '//div[@class="input"][@dir="auto"][@data-tab="1"]'
input_box = wait.until(EC.presence_of_element_located(( 
	By.XPATH, inp_xpath))) 
for i in range(100): 
	input_box.send_keys(string + Keys.ENTER) 
	time.sleep(1) 
