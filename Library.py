import numpy as np
import pandas as pd

BOUND_NAMES = ['ch_max', 'ch_min', 'dis_max', 'dis_min', 'ene_max', 'ene_min']


def update_df(df1, df2):
    """ df1을 df2를 기반으로 업데이트하는 함수
    새로운 열이 df1에 없으면 추가하거나 기존 열에 해당하는 인덱스에 요소를 삽입함
    여기서 df1은 편집할 원래 데이터프레임, df2는 추가할 데이터 프레임을 의미하며
    업데이트된 데이터프레임인 df1을 반환하게 됨
    """

    old_col = set(df2.columns).intersection(set(df1.columns))
    df1 = df1.join(df2[list(set(df2.columns).difference(old_col))], how='left')  # join new columns
    df1.update(df2[list(old_col)])  # update old columns
    return df1


def disagg_col(df, group, col):
    """ group의 카운트를 기반으로 col의 비계층화된 열을 추가함
    여기서 group은 리스트로 그룹화할 열, col은 문자열로 비계층화할 열을 의미하며
    비계층화된 열이 추가되어 업데이트된 데이터프레임인 df를 반환함
    """
    count_df = df.groupby(by=group).size()
    count_df.name = 'counts'
    df = df.reset_index().merge(count_df.reset_index(), on=group, how='left').set_index(df.index.names)
    df[col+'_disagg'] = df[col] / df['counts']
    return df


def apply_growth(source, rate, source_year, yr, freq):
    """ 선형 성장률을 적용하여 미래 연도의 데이터를 결정하는 함수
    여기서 source는 시리즈로 주어진 데이터, rate는 부동소수점으로 연간 성장률(%),
    source_year는 기간으로 주어진 데이터 연도, yr는 기간으로 데이터를 가져올 미래 연도,
    freq는 문자열로 시뮬레이션 타임 스텝 주기를 의미함
    미래 연도의 데이터가 포함된 새로운 시리즈를 반환함
    """
    years = yr.year - source_year.year  # difference in years between source and desired yea
    new = source*(1+rate)**years  # apply growth rate to source data
    # new.index = new.index + pd.DateOffset(years=1)
    # deal with leap years
    source_leap = is_leap_yr(source_year.year)
    new_leap = is_leap_yr(yr.year)

    if (not source_leap) and new_leap:   # need to add leap day
        # if source is not leap year but desired year is, copy data from previous day
        new.index = new.index + pd.DateOffset(years=years)
        leap_ind = pd.date_range(start='02/29/'+str(yr), end='03/01/'+str(yr), freq=freq, closed='left')
        leap = pd.Series(new[leap_ind - pd.DateOffset(days=1)].values, index=leap_ind, name=new.name)
        new = pd.concat([new, leap])
        new = new.sort_index()
    elif source_leap and (not new_leap):  # need to remove leap day
        leap_ind = pd.date_range(start='02/29/'+str(source_year), end='03/01/'+str(source_year), freq=freq, closed='left')
        new = new[~new.index.isin(leap_ind)]
        new.index = new.index + pd.DateOffset(years=years)
    else:
        new.index = new.index + pd.DateOffset(years=years)
    return new


def create_timeseries_index(years, frequency):
    """ 시간 색인의 템플릿을 만들며 내부 프로그램에서 시간 단위 시작을 나타내는 함수
    years는 리스트로 포함된 연도 목록을, frequency는 문자열로 판다스 주기의 문자열을 표현하며
    날짜 범위를 만드는 데 필요함
    시간 색인이 0시에서 시작하는 빈 데이터프레임을 반환함
    """
    temp_master_df = pd.DataFrame()
    years = np.sort(years)
    for year in years:
        new_index = pd.date_range(start=f"1/1/{int(year)}", end=f"1/1/{int(year + 1)}", freq=frequency, closed='left')
        temp_df = pd.DataFrame(index=new_index)

        # add new year to original data frame
        temp_master_df = pd.concat([temp_master_df, temp_df], sort=True)
    temp_master_df.index.name = 'Start Datetime (hb)'
    return temp_master_df.index


def fill_extra_data(df, years_need_data_for, growth_rate, frequency):
    """ 주어진 성장률로 추정된 누락된 연도의 데이터로 시계열 데이터를 확장하는 함수
    df는 다음에 적용할 데이터프레임, years_need_data_for는 데이터프레임의 색인에 포함되어야 하는 연도,
    growth_rate는 시간이 지남에 따라 데이터가 성장하는 속도, frequency는 판다스 주기의 문자열을 표현함
    업데이트된 데이터프레임인 df를 반환함함
    """
    data_year = df.iloc[1:].index.year.unique()  # grab all but the first index
    # which years do we not have data for
    no_data_year = {pd.Period(year) for year in years_need_data_for} - {pd.Period(year) for year in data_year}
    # if there is a year we dont have data for
    if len(no_data_year) > 0:
        for yr in no_data_year:
            source_year = pd.Period(max(data_year))  # which year to to apply growth rate to (is this the logic we want??)
            source_data = df.loc[df.index.year == source_year.year]  # use source year data

            # create new dataframe for missing year
            try:
                new_data_df = pd.DataFrame()
                for col in df.columns:
                    new_data = apply_growth(source_data[col], growth_rate, source_year, yr, frequency)  # apply growth rate to column
                    new_data_df = pd.concat([new_data_df, new_data], axis=1, sort=True)
                    # add new year to original data frame
                df = pd.concat([df, new_data_df], sort=True)
            except AttributeError:
                new_data = apply_growth(source_data, growth_rate, source_year, yr, frequency)  # apply growth rate to column
                # add new year to original data frame
                df = pd.concat([df, new_data], sort=True)
    return df


def drop_extra_data(df, years_need_data_for):
    """ years_need_data_for에 지정되지 않은 데이터를 제거하는 함수
    지정된 연도의 데이터만 포함된 데이터프레임을 반환함
    """
    data_year = df.index.year.unique()  # which years was data given for
    # which years is data given for that is not needed
    dont_need_year = {pd.Period(year) for year in data_year} - {pd.Period(year) for year in years_need_data_for}
    if len(dont_need_year) > 0:
        for yr in dont_need_year:
            df_sub = df.loc[df.index.year != yr.year]  # choose all data that is not in the unneeded year
            df = df_sub
    return df


def is_leap_yr(year):
    """ 주어진 연도가 윤년인지 여부를 결정하는 함수
    윤년이면 True, 아니면 False를 반환함
    """
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0


def truncate_float(number, decimals=3):
    """ 부동소수점을 지정된 소수점 자릿수로 절사하는 함수
    nimber는 많은 소수 자릿수를 갖는 부동소수점, decimals는 유지할 소수점 자릿수를 의미함
    입력 숫자의 절사된 버전을 반환함
    """
    return round(number * 10**decimals) / 10**decimals
