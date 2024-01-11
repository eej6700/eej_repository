"""SystemRequirement.py

This file hold 2 classes: Requirements and System Requirements
"""

import storagevet.Library as Lib
import numpy as np
import pandas as pd


VERY_LARGE_NUMBER = 2**32 - 1
VERY_LARGE_NEGATIVE_NUMBER = -1 * VERY_LARGE_NUMBER


class Requirement:
    """ This class creates a requirement that needs to be meet (regardless of what other requirements exist)

    """

    def __init__(self, constraint_type, limit_type, parent, constraint):
        """ Requirement 객체를 초기화하고 해당 값(value)을 설정하는 함수
        constraint_type: 에너지, 충전 또는 방전 상태 (대소문자 구분 없음)
        limit_type: 최대 또는 최소 (대소문자 구분 없음, 'max' 또는 'min'이어야 함)
        parent: 제약 조건을 생성한 "service" 또는 "technology"를 나타냄
        constraint: 제약 조건 값; 물리적 제약 조건인 경우 float, 제어 또는 서비스 제약 조건인 경우 Timestamp 인덱스가 있는 Series
        """
        self.type = constraint_type.lower()
        self.limit_type = limit_type.lower()
        self.value = constraint
        self.parent = parent


class SystemRequirement:
    """ This class is meant to handle Requirements of the same type. Determines what min/max value (as a function of time) would
    ensure all of the requirements are met. Hold information about what Value Stream(s) set the value and which others have "contributed"

    """

    def __init__(self, constraint_type, limit_type, years_of_analysis, datetime_freq):
        """ Constraint을 초기화하는 함수
        constraint_type: 에너지, 충전 또는 방전 상태 (대소문자 구분 없음)
        limit_type: 최대 또는 최소 (대소문자 구분 없음)
        years_of_analysis: 반환된 인덱스에 포함되어야 하는 연도 목록
        datetime_freq: Pandas 빈도를 나타내는 문자열 표현입니다. DateTime 범위를 만드는 데 필요
        """
        self.type = constraint_type.lower()
        self.is_max = limit_type.lower() == 'max'
        self.is_min = limit_type.lower() == 'min'
        if limit_type not in ['max', 'min']:
            raise SyntaxWarning("limit_type can be 'max' or 'min'")

        index = Lib.create_timeseries_index(years_of_analysis, datetime_freq)
        size = len(index)
        self.parents = pd.DataFrame(columns=['DateTime', 'Parent'])  # records which valuestreams have had a non-base value
        self.owner = pd.Series(np.repeat('null', size), index=index)  # records which valuestream(s) set the current VALUE

        # records the value that would ensure all past updated requirements would also be met
        #   this is needed because of the way that system requirements are created.
        #   you start with all huge or hugely negative values and then values become updated
        #   the update() method is used to build values into these requirements
        #   and lets you handle multiple system requirements of the same type,
        #   without creating a new constraint for each one.
        if self.is_min:
            self.value = pd.Series(np.repeat(VERY_LARGE_NEGATIVE_NUMBER, size), index=index)
        if self.is_max:
            self.value = pd.Series(np.repeat(VERY_LARGE_NUMBER, size), index=index)

    def update(self, requirement):
        """ 제약 조건 값을 업데이트하고 각 타임스탬프에 기여하는 ValueStream을 기록하는 함수수
        """
        parent = requirement.parent
        value = requirement.value
        update_indx = value.index

        # record the timestamps and parent
        temp_df = pd.DataFrame({'DateTime': update_indx.values})
        temp_df['Parent'] = parent
        self.parents = self.parents.append(temp_df, ignore_index=True)

        # check whether the value needs to be updated, if so--then also update the owner value
        update_values = self.value.loc[update_indx]
        if self.is_min:
            # if minimum constraint, choose higher constraint value
            new_constraint = np.maximum(value.values, update_values.values)
        else:
            # if maximum constraint, choose lower constraint value
            new_constraint = np.minimum(value.values, update_values.values)
        self.value.loc[update_indx] = new_constraint
        # figure out which values changed, and at which indexes
        mask = update_values != new_constraint

        # update the owner at the indexes found above
        self.owner[update_indx][mask] = parent

    def contributors(self, datetime_indx):
        """ 지정된 시간 동안 제약 조건의 부모를 가져오는 함수
        datetime_indx : 고려해야 하는 타임스탬프의 데이터 시간 인덱스
        """
        contributors = self.parents[self.parents['DateTime'].isin(datetime_indx.to_list())].Parent
        return contributors.unique()

    def get_subset(self, mask):
        """ 주어진 mask에 해당하는 시간에 대한 이 요구 사항의 값을 가져오는 함수
        """
        return self.value.loc[mask].values

    def __le__(self, other):
        """ 비교 연산자 오버로딩 메서드로, 다른 SystemReqirement 객체 또는 정수와의 비교를 처리하는 함수
        """
        try:
            return self.value <= other.value
        except AttributeError:
            return self.value <= other

    def __lt__(self, other):
        """ 비교 연산자 오버로딩 메서드로, 다른 SystemReqirement 객체 또는 정수와의 비교를 처리하는 함수
        """
        try:
            return self.value < other.value
        except AttributeError:
            return self.value < other

    def __eq__(self, other):
        """ 비교 연산자 오버로딩 메서드로, 다른 SystemReqirement 객체 또는 정수와의 비교를 처리하는 함수
        """
        try:
            return self.value == other.value
        except AttributeError:
            return self.value == other

    def __ne__(self, other):
         """ 비교 연산자 오버로딩 메서드로, 다른 SystemReqirement 객체 또는 정수와의 비교를 처리하는 함수
        """
        try:
            return self.value != other.value
        except AttributeError:
            return self.value != other

    def __gt__(self, other):
        """ 비교 연산자 오버로딩 메서드로, 다른 SystemReqirement 객체 또는 정수와의 비교를 처리하는 함수
        """
        try:
            return self.value > other.value
        except AttributeError:
            return self.value > other

    def __ge__(self, other):
        """ 비교 연산자 오버로딩 메서드로, 다른 SystemReqirement 객체 또는 정수와의 비교를 처리하는 함수
        """
        try:
            return self.value >= other.value
        except AttributeError:
            return self.value >= other
