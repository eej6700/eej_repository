"""
Copyright (c) 2023, Electric Power Research Institute

 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import pandas as pd
import numpy as np
import cvxpy as cvx
from storagevet.SystemRequirement import SystemRequirement
from storagevet.ErrorHandling import *


class ServiceAggregator:
    """ Tracks the value streams and bids the storage's capabilities into markets
    """

    def __init__(self, value_stream_inputs_map, value_stream_class_map):
        """ 클래스의 초기화 함수
        서비스 집합 및 관련 매개변수를 초기화함
        value_stream_inputs_map: 활성화된 모든 서비스에 대한 입력 매핑 정보를 포함하는 딕셔너리
        value_stream_class_map: 서비스 이름과 해당 서비스 클래스를 연결하는 딕셔너리
        """

        self.value_streams = {}

        for service, params_input in value_stream_inputs_map.items():
            if params_input is not None:  # then Params class found an input
                TellUser.info("Initializing: " + str(service))
                self.value_streams[service] = value_stream_class_map[service](params_input)
        TellUser.debug("Finished adding value streams")

        self.sys_requirements = {}
        self.system_requirements_conflict = False

    def update_analysis_years(self, end_year, poi, frequency, opt_years, def_load_growth):
        """ 분석 연도를 업데이트하고 특정 서비스에서 사용하는 함수
        연도 간의 변화 및 시스템의 노출 검사
        """
        if 'Deferral' in self.value_streams.keys():
            return self.value_streams['Deferral'].check_for_deferral_failure(end_year, poi, frequency, opt_years, def_load_growth)
        return opt_years

    def is_deferral_only(self):
        """ Deferral 서비스만 사용 중이면 최적화를 건너뛰도록 하는 함수
        """
        return 'Deferral' in self.value_streams and len(self.value_streams) == 1

    def initialize_optimization_variables(self, size):
        """ 최적화 변수를 초기화하는 함수
        """
        for value_stream in self.value_streams.values():
            value_stream.initialize_variables(size)

    def identify_system_requirements(self, der_lst, years_of_analysis, datetime_freq):
        """ 시스템 요구 사항을 확인하고 시스템 제약 조건을 설정하는 함수수
        """
        for service in self.value_streams.values():
            service.calculate_system_requirements(der_lst)
            for constraint in service.system_requirements:
                # check to see if system requirement has been initialized
                limit_key = f"{constraint.type} {constraint.limit_type}"
                sys_requ = self.sys_requirements.get(limit_key)
                if sys_requ is None:
                    # if not, then initialize one
                    sys_requ = SystemRequirement(constraint.type, constraint.limit_type, years_of_analysis, datetime_freq)
                # update system requirement
                sys_requ.update(constraint)
                # save system requirement
                self.sys_requirements[limit_key] = sys_requ

        # report the datetimes and VS that contributed to the conflict
        # 1) check if poi import max and poi import min conflict
        if self.sys_requirements.get('poi import min') is not None and self.sys_requirements.get('poi import max') is not None:
            poi_import_conflicts = self.sys_requirements.get('poi import min') > self.sys_requirements.get('poi import max')
            self.report_conflict(poi_import_conflicts, ['poi import min', 'poi import max'])
        # 2) check if energy max and energy min conflict
        if self.sys_requirements.get('energy min') is not None and self.sys_requirements.get('energy max') is not None:
            energy_conflicts = self.sys_requirements.get('energy min') > self.sys_requirements.get('energy max')
            self.report_conflict(energy_conflicts, ['energy min', 'energy max'])
        # 3) check if poi export max and poi export min conflict
        if self.sys_requirements.get('poi export min') is not None and self.sys_requirements.get('poi export max') is not None:
            poi_export_conflicts = self.sys_requirements.get('poi export min') > self.sys_requirements.get('poi export max')
            self.report_conflict(poi_export_conflicts, ['poi export min', 'poi export max'])
        # 4) check if poi export min and poi import min conflict (cannot be > 0 (nonzero at input) at the same time)
        if self.sys_requirements.get('poi import min') is not None and self.sys_requirements.get('poi export min') is not None:
            poi_import_and_poi_export_conflicts = (self.sys_requirements.get('poi import min') > 0) & (self.sys_requirements.get('poi export min') > 0)
            self.report_conflict(poi_import_and_poi_export_conflicts, ['poi import min', 'poi export min'])
        if self.system_requirements_conflict:
            raise SystemRequirementsError('System requirements are not possible. Check log files for more information.')
        else:
            return self.sys_requirements

    def report_conflict(self, conflict_mask, check_sys_req):
        """ 시스템 요구 사항에 충돌이 있는지 확인하고 충돌이 있을 경우 사용자에게 보고하며
        최적화가 불가능하다고 플래그를 설정하여 실행을 중지하는 함수수
        """
        if np.any(conflict_mask):
            self.system_requirements_conflict = True
            datetimes = conflict_mask.index[conflict_mask]
            if len(datetimes):
                TellUser.error(f'System requirements are not possible at {datetimes.to_list()}')
                for req in check_sys_req:
                    TellUser.error(f"The following contribute to the {req} error: {self.sys_requirements.get(req).contributors(datetimes)}")

    def optimization_problem(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating, annuity_scalar=1):
        """ 최적화 문제를 정의하고 목적 함수 및 제약 조건을 반환하는 함수
        """
        opt_functions = {}
        opt_constraints = []
        for value_stream in self.value_streams.values():
            opt_functions.update(value_stream.objective_function(mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, annuity_scalar))
            opt_constraints += value_stream.constraints(mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating)
        return opt_functions, opt_constraints

    def aggregate_reservations(self, mask):
        """ 전체 전력 예약을 계산하는 함수
        서비스별로 충전 및 방전 예약을 집계하는 함수수
        """
        charge_up = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        charge_down = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        discharge_up = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        discharge_down = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        uenergy_stored = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        uenergy_provided = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        worst_ue_stored = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        worst_ue_provided = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')

        for value_stream in self.value_streams.values():
            charge_up += value_stream.p_reservation_charge_up(mask)
            charge_down += value_stream.p_reservation_charge_down(mask)
            discharge_up += value_stream.p_reservation_discharge_up(mask)
            discharge_down += value_stream.p_reservation_discharge_down(mask)
            uenergy_stored += value_stream.uenergy_option_stored(mask)
            uenergy_provided += value_stream.uenergy_option_provided(mask)
            worst_ue_stored += value_stream.worst_case_uenergy_stored(mask)
            worst_ue_provided += value_stream.worst_case_uenergy_provided(mask)

        return discharge_down, discharge_up, charge_down, charge_up, uenergy_provided, uenergy_stored, worst_ue_provided, worst_ue_stored

    def save_optimization_results(self, subs_index):
        """ 최적화 결과를 저장하는 함수
        각 서비스에 대한 최적화 결과를 저장함
        """
        for value_stream in self.value_streams.values():
            value_stream.save_variable_results(subs_index)

    def merge_reports(self):
        """ 모든 서비스의 최적화 결과를 수집하고 병합하여 사용자 친화적인 형태로 반환하는 함수
        """
        results = pd.DataFrame()
        monthly_data = pd.DataFrame()

        for service in self.value_streams.values():
            report_df = service.timeseries_report()
            results = pd.concat([results, report_df], axis=1, sort=False)
            report = service.monthly_report()
            monthly_data = pd.concat([monthly_data, report], axis=1, sort=False)
        return results, monthly_data

    def drill_down_dfs(self, **kwargs):
        """ 결과를 자세히 분석하기 위한 데이터프레임을 반환하는 함수
        각 서비스에 대한 드릴다운 보고서를 생성함
        """
        df_dict = dict()
        for der in self.value_streams.values():
            df_dict.update(der.drill_down_reports(**kwargs))
        return df_dict
