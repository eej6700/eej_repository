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
"""
POI.py

"""
import numpy as np
import cvxpy as cvx
import pandas as pd
from storagevet.ErrorHandling import *


class POI:
    # 에너지 시스템의 포인트 오브 인터커넥션을 나타내는 클래스

    def __init__(self, params, technology_inputs_map, technology_class_map):
        """ POI 객체를 초기화하하는 함수
        사용자가 정의한 모델 매개변수 및 기술 입력을 사용하여 DER 객체들을 초기화하고 활성 DER 목록에 추가함
        """
        self.max_export = params['max_export']  # > 0
        self.max_import = params['max_import']  # < 0
        self.apply_poi_constraints = params['apply_poi_constraints']

        # types of DER technology, and their dictionary of active specific technologies
        self.der_list = []
        self.der_summary = {}  # keys= names of DERs, values= DER type  (basically tech summary output)
        self.active_ders = []

        # initialize all DERs
        for der, params_input in technology_inputs_map.items():
            if params_input is not None:  # then Params class found an input
                for id_val, id_der_input in params_input.items():
                    # add ID to der input dictionary
                    id_der_input['ID'] = id_val
                    TellUser.info(f"Initializing: {der}-{id_val}")
                    der_object = technology_class_map[der](id_der_input)
                    self.der_list.append(der_object)
                    self.active_ders.append(der_object)

    def calculate_system_size(self):
        """ 시스템이 추출할 수 있는 최대 전력, 수입할 수 있는 최대 전력, 저장할 수 있는 최대 에너지 및 최소 에너지를 계산함
        """
        min_ene = 0
        max_ene = 0
        max_ch = 0
        max_dis = 0

        for der_instance in self.active_ders:
            min_ene += der_instance.operational_min_energy()
            max_ene += der_instance.operational_max_energy()
            max_ch += der_instance.charge_capacity()
            max_dis += der_instance.discharge_capacity()
        return max_ch, max_dis, (max_ene, min_ene)

    def initialize_optimization_variables(self, size):
        """ 최적화 루프의 시작에서 최적화 변수를 초기화하는 함수 
        """
        for der_instance in self.active_ders:
            # initialize variables
            der_instance.initialize_variables(size)

    def get_state_of_system(self, mask):
        """ POI의 상태를 측정하여 최적화에 사용될 파라미터를 계산하는 함수수
        """
        opt_var_size = sum(mask)
        load_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        var_gen_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        gen_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        tot_net_ess = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        der_dispatch_net_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        total_soe = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        agg_power_flows_in = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        agg_power_flows_out = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI

        agg_steam_heating_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        agg_hotwater_heating_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        agg_thermal_cooling_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI

        for der_instance in self.active_ders:
            # add the state of the der's power over time & stored energy over time to system's
            # these agg_power variables are used with POI constraints
            agg_power_flows_in += (der_instance.get_charge(mask) - der_instance.get_discharge(mask))
            agg_power_flows_out += (der_instance.get_discharge(mask) - der_instance.get_charge(mask))

            if der_instance.technology_type == 'Load':
                load_sum += der_instance.get_charge(mask)
            if der_instance.technology_type == 'Energy Storage System':
                total_soe += der_instance.get_state_of_energy(mask)
                tot_net_ess += der_instance.get_net_power(mask)
                # auxiliary load (hp) for a Battery will contribute to agg_power variables
                #   in a similar manner to SiteLoad (add to flow_in and subtract from flow_out)
                try:
                    aux_load = der_instance.hp
                    agg_power_flows_in += aux_load
                    agg_power_flows_out -= aux_load
                except AttributeError:
                    pass
            if der_instance.technology_type == 'Generator':
                gen_sum += der_instance.get_discharge(mask)
            if der_instance.technology_type == 'Intermittent Resource':
                var_gen_sum += der_instance.get_discharge(mask)
            if der_instance.technology_type in ['Energy Storage System', 'Generator']:
                der_dispatch_net_power += der_instance.get_net_power(mask)

        ##NOTE: these print statements disclose info for get_state_of_system cvx Parameters
        #print(f'load_sum:               ({load_sum})')
        #print(f'var_gen_sum:            ({var_gen_sum})')
        #print(f'gen_sum:                ({gen_sum})')
        #print(f'tot_net_ess:            ({tot_net_ess})')
        #print(f'der_dispatch_net_power: ({der_dispatch_net_power})')
        #print(f'total_soe:              ({total_soe})')
        #print(f'agg_power_flows_in:     ({agg_power_flows_in.name()})')
        #print(f'agg_power_flows_out:    ({agg_power_flows_out.name()})')

        return load_sum, var_gen_sum, gen_sum, tot_net_ess, der_dispatch_net_power, total_soe, agg_power_flows_in, agg_power_flows_out, agg_steam_heating_power, agg_hotwater_heating_power, agg_thermal_cooling_power

    def combined_discharge_rating_for_reliability(self):
        """ 신뢰성 제약을 위한 복합 방전 등급을 계산하는 함수수
        """
        combined_rating = 0
        for der_instance in self.active_ders:
            if der_instance.technology_type == 'Energy Storage System':
                combined_rating += der_instance.dis_max_rated
            if der_instance.technology_type == 'Generator':
                combined_rating += der_instance.rated_power
        return combined_rating

    def optimization_problem(self, mask, power_in, power_out, steam_in, hotwater_in, cold_in, annuity_scalar=1):
        """ 최적화 문제를 생성하고 POI의 제약 조건 리스트를 반환하는 함수수
        """
        constraint_list = []
        opt_size = sum(mask)
        obj_expression = {}  # dict of objective costs

        # deal with grid_charge constraint btw ESS and PV
        total_pv_out_ess_can_charge_from = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        total_ess_charge = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        allow_charge_from_grid = True
        # deal with inverter constraint btw ess and any PV marked as dc coupled
        total_pv_out_dc = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        net_ess_power = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        agg_inv_max = 0
        dc_coupled_pvs = False

        for der_instance in self.active_ders:
            # add all operational constraints
            constraint_list += der_instance.constraints(mask)
            # add DER cost funcs
            obj_expression.update(der_instance.objective_function(mask, annuity_scalar))
            if der_instance.tag == 'PV':
                if not der_instance.grid_charge:
                    allow_charge_from_grid = False
                    total_pv_out_ess_can_charge_from += der_instance.get_discharge(mask)
                if der_instance.loc == 'dc':
                    dc_coupled_pvs = True
                    total_pv_out_dc += der_instance.get_discharge(mask)
                    agg_inv_max += der_instance.inv_max
            if der_instance.technology_type == 'Energy Storage System':
                net_ess_power += der_instance.get_net_power(mask)
                total_ess_charge += der_instance.get_charge(mask)

        if not allow_charge_from_grid:  # add grid charge constraint
            constraint_list += [cvx.NonPos(total_ess_charge - total_pv_out_ess_can_charge_from)]

        if dc_coupled_pvs:  # add dc coupling constraints
            constraint_list += [cvx.NonPos(total_pv_out_dc + (-1) * net_ess_power - agg_inv_max)]
            constraint_list += [cvx.NonPos(-agg_inv_max - total_pv_out_dc + net_ess_power)]

        # power import/export constraints
        # NOTE: power_in is agg_p_in (defined in self.get_state_of_system)
        # NOTE: power_out is agg_p_out (defined in self.get_state_of_system)
        if self.apply_poi_constraints:
            # (agg_p_in) <= -max_import
            constraint_list += [cvx.NonPos(self.max_import + power_in)]

            # (agg_p_out) <= max_export
            # NOTE: with active_load_dump True, we do not include this constraint
            #   active_load_dump only occurs in DER-VET
            if not self.disable_max_export_poi_constraint():
                constraint_list += [cvx.NonPos(power_out + -1 * self.max_export)]

        return obj_expression, constraint_list

    def disable_max_export_poi_constraint(self):
        # 최대 수출 POI 제약을 비활성화하는지 여부를 반환하는 함수수
        try:
            disable_max_export = self.active_load_dump
            if disable_max_export:
                TellUser.warning(f'You have activated a load dump in DER-VET, therefore we disable the max_export POI constraint in the optimization. The load dump will be determined as a post-optimization calculation.')
            return disable_max_export
        except AttributeError:
            return False

    def aggregate_p_schedules(self, mask):
        """ DER의 방전 및 충전 일정을 합산하여 전력 조정 서비스에 참여할 수 있는 능력을 계산하는 함수수
        """
        opt_size = sum(mask)
        agg_dis_up = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        agg_dis_down = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        agg_ch_up = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        agg_ch_down = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        uenergy_incr = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        uenergy_decr = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        uenergy_thru = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')

        for der_in_market_participation in self.active_ders:
            if der_in_market_participation.can_participate_in_market_services:
                agg_ch_up += der_in_market_participation.get_charge_up_schedule(mask)
                agg_ch_down += der_in_market_participation.get_charge_down_schedule(mask)
                agg_dis_up += der_in_market_participation.get_discharge_up_schedule(mask)
                agg_dis_down += der_in_market_participation.get_discharge_down_schedule(mask)
                uenergy_incr += der_in_market_participation.get_uenergy_increase(mask)
                uenergy_decr += der_in_market_participation.get_uenergy_decrease(mask)
                uenergy_thru += der_in_market_participation.get_delta_uenegy(mask)

        return agg_dis_down, agg_dis_up, agg_ch_down, agg_ch_up, uenergy_decr, uenergy_incr, uenergy_thru

    def merge_reports(self, is_dispatch_opt, index):
        """ DER들의 최적화 결과를 수집하고 병합하여 사용자 친화적인 결과를 나타내는 데이터프레임을 반환하는 함수수
        """
        results = pd.DataFrame(index=index)
        monthly_data = pd.DataFrame()

        # initialize all the data columns that will ALWAYS be present in our results
        results.loc[:, 'Total Load (kW)'] = 0
        results.loc[:, 'Total Generation (kW)'] = 0
        results.loc[:, 'Total Storage Power (kW)'] = 0
        results.loc[:, 'Aggregated State of Energy (kWh)'] = 0

        for der_instance in self.der_list:
            report_df = der_instance.timeseries_report()
            results = pd.concat([report_df, results], axis=1)
            if der_instance.technology_type in ['Generator', 'Intermittent Resource']:
                results.loc[:, 'Total Generation (kW)'] += results[f'{der_instance.unique_tech_id()} Electric Generation (kW)']
            if der_instance.technology_type == 'Energy Storage System':
                results.loc[:, 'Total Storage Power (kW)'] += results[f'{der_instance.unique_tech_id()} Power (kW)']
                results.loc[:, 'Aggregated State of Energy (kWh)'] += results[f'{der_instance.unique_tech_id()} State of Energy (kWh)']
                # add any battery auxiliary load (hp) to the total load
                try:
                    auxiliary_load = der_instance.hp
                    if auxiliary_load > 0:
                        TellUser.info(f'adding auxiliary load ({auxiliary_load} from ESS: {der_instance.name} to the Total Load')
                        results.loc[:, 'Total Load (kW)'] += auxiliary_load
                except AttributeError:
                    pass
            if der_instance.technology_type == 'Load':
                results.loc[:, 'Total Load (kW)'] += results[f'{der_instance.unique_tech_id()} Original Load (kW)']
            report = der_instance.monthly_report()
            monthly_data = pd.concat([monthly_data, report], axis=1, sort=False)

        # assumes the orginal net load only does not contain the Storage system
        # net load is the load see at the POI
        results.loc[:, 'Net Load (kW)'] = results.loc[:, 'Total Load (kW)'] - results.loc[:, 'Total Generation (kW)'] - results.loc[:, 'Total Storage Power (kW)']
        return results, monthly_data

    def technology_summary(self):
        """ DER들의 종류 및 이름에 대한 데이터프레임을 생성하여 반환하는 함수
        """
        der_summary = {'Type': [], 'Name': []}
        for der_object in self.der_list:
            der_summary['Type'].append(der_object.technology_type)
            der_summary['Name'].append(der_object.name)
        technology_summary = pd.DataFrame(der_summary)
        technology_summary.set_index('Name')
        return technology_summary

    def drill_down_dfs(self, **kwargs):
        """ 결과를 세부적으로 분석하기 위한 데이터프레임을 반환하는 함수
        """
        df_dict = dict()
        for der in self.der_list:
            df_dict.update(der.drill_down_reports(**kwargs))
        return df_dict
