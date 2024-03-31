from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import FirefoxProfile, Firefox
import pandas as pd

wunderground_data_path = 'Wunderground Data/'


def prepare_firefox_driver():
    gecko_path = "C:\Users\Abrar Abir\Documents\CPTS_591\Project\geckodriver.exe"
    profile = FirefoxProfile()
    profile.set_preference('general.warnOnAboutConfig', False)
    profile.update_preferences()
    return Firefox(firefox_profile=profile, executable_path=gecko_path)


if __name__ == '__main__':
    regions = ['San_Francisco', 'Austin', 'Seattle',
               'Phoenix', 'Denver', 'Salt_Lake_City',
               'Portland', 'Las_Vegas', 'Boise',
               'Albuquerque', 'Billings', 'Cheyenne'][6] + '/'
    region_url_key = ['ca/san-francisco/KSFO', 'tx/austin/KAUS', 'wa/seattle/KSEA',
                      'az/phoenix/KPHX', 'co/denver/KDEN', 'ut/salt-lake-city/KSLC',
                      'or/portland/KPDX', 'nv/las-vegas/KVGT', 'id/boise/KBOI',
                      'nm/albuquerque/KABQ', 'mt/billings/KBIL', 'wy/cheyenne/KCYS'][6]

    viewButton = '//*[@id="inner-content"]/div[2]/div[1]/div[1]/div[1]/div/lib-date-selector/div/input'
    tableElem = '//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table'

    temp_columns, string_columns = ['Temperature', 'Dew Point'], ['Time', 'Wind', 'Condition']

    driver = prepare_firefox_driver()
    timeRange = pd.date_range('2012-01-01', '2016-12-31')

    for single_date in timeRange:
        print(single_date)
        url = f'https://www.wunderground.com/history/daily/{region_url_key}/date/' + str(single_date.date())
        driver.get(url)
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, viewButton))).click()

        table = WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.XPATH, tableElem)))
        headers = [header.text for header in table.find_elements_by_xpath(".//tr")[0].find_elements_by_xpath(".//th")]
        bodyInfo = [[cell.text for cell in row.find_elements_by_xpath(".//td")] for row in
                    table.find_elements_by_xpath(".//tr")[1:]]

        df = pd.DataFrame(data=bodyInfo, columns=headers).set_index('Time')
        df.index = pd.to_datetime(str(single_date.date()) + ' ' + df.index)
        df[df.columns.difference(string_columns)] = df[df.columns.difference(string_columns)].apply(
            lambda x: x.str.split().str[0].astype('float'))
        df[temp_columns] = df[temp_columns].apply(lambda x: ((x - 32) * 5 / 9).round(2))

        save_path = wunderground_data_path + 'USA/' + regions + str(single_date.date())  # usa
        df.to_csv(save_path)
