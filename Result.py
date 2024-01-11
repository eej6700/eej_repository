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
Result.py
# 최적화 결과를 수집하고 저장하며, 후속 계산 및 CSV 파일로 저장하는 클래스
"""
import pandas as pd
from storagevet.ErrorHandling import *


class Result:
    """ This class serves as the later half of DER-VET's 'case builder'. It collects all optimization results, preforms
    any post optimization calculations, and saves those results to disk. If there are multiple

    """
    # these variables get read in upon importing this module
    instances = None
    sensitivity_df = None
    sensitivity = False
    dir_abs_path = None

    @classmethod
    def initialize(cls, results_params, case_definitions):
        """ 클래스의 속성을 초기화하고 여러 최적화 실행을 비교할 수 있도록 결과 인스턴스를 생성하는 함수
        results_params : 모델 파라미터 입력에서 사용자가 정의한 값들을 담은 딕셔너리
        case_definitions : 감도 분석 인스턴스를 담은 데이터프레임임
        """
        cls.instances = {}
        cls.dir_abs_path = Path(results_params['dir_absolute_path'])
        cls.csv_label = results_params.get('label', '')
        if cls.csv_label == 'nan':
            cls.csv_label = ''
        cls.sensitivity_df = case_definitions

        # data frame of all the sensitivity instances
        cls.sensitivity = (not cls.sensitivity_df.empty)
        if cls.sensitivity:
            # edit the column names of the sensitivity df to be human readable
            human_readable_names = [f"[SP] {col_name[0]} {col_name[0]}" for col_name in cls.sensitivity_df.columns]
            # TODO: use scehma.xml to get unit of tag-key combo and add to HUMAN_READABLE_NAMES
            cls.sensitivity_df.columns = human_readable_names
            cls.sensitivity_df.index.name = 'Case Number'

    @classmethod
    def add_instance(cls, key, scenario):
        """ 결과 인스턴스를 생성하고 딕셔너리에 추가하는 함수
        key: 결과 인스턴스가 나타내는 Params 클래스의 인스턴스 값에 대응하는 키
        scenario : 최적화가 완료된 후의 시나리오 객체체
        """
        # initialize an instance of Results
        template = cls(scenario)
        # save it in our dictionary of instance (so we can keep track of all the Results we have)
        cls.instances.update({key: template})
        # preform post facto calculations and CBA
        template.collect_results()
        template.create_drill_down_dfs()
        template.calculate_cba()
        # save dataframes as CSVs
        template.save_as_csv(key, cls.sensitivity)

    def __init__(self, scenario):
        """ Result 객체를 초기화하는 함수
        scenario : 최적화가 완료된 후의 시나리오 객체를 받아 초기화함함
        """
        self.frequency = scenario.frequency
        self.dt = scenario.dt
        self.verbose_opt = scenario.verbose_opt
        self.n = scenario.n
        self.n_control = scenario.n_control
        self.mpc = scenario.mpc
        self.start_year = scenario.start_year
        self.end_year = scenario.end_year
        self.opt_years = scenario.opt_years
        self.incl_binary = scenario.incl_binary
        self.incl_slack = scenario.incl_slack
        self.verbose = scenario.verbose
        self.poi = scenario.poi
        self.service_agg = scenario.service_agg
        self.objective_values = scenario.objective_values
        self.cost_benefit_analysis = scenario.cost_benefit_analysis
        self.opt_engine = scenario.opt_engine

        # initialize DataFrames that drill down dfs will be built off
        self.time_series_data = pd.DataFrame(index=scenario.optimization_levels.index)
        self.monthly_data = pd.DataFrame()
        self.technology_summary = pd.DataFrame()

        # initialize the dictionary that will hold all the drill down plots
        self.drill_down_dict = dict()

    def collect_results(self):
        """ 최적화 변수 솔루션이나 사용자 입력을 수집하여 드릴다운 플롯 및 사용자에게 보고할 데이터프레임을 생성하는 함수
        """

        TellUser.debug("Performing Post Optimization Analysis...")

        report_df, monthly_report = self.poi.merge_reports(self.opt_engine,
                                                           self.time_series_data.index)
        self.time_series_data = pd.concat([self.time_series_data, report_df],
                                          axis=1)
        self.monthly_data = pd.concat([self.monthly_data, monthly_report],
                                      axis=1, sort=False)

        # collect results from each value stream
        ts_df, month_df = self.service_agg.merge_reports()
        self.time_series_data = pd.concat([self.time_series_data, ts_df], axis=1)

        self.monthly_data = pd.concat([self.monthly_data, month_df], axis=1, sort=False)

        self.technology_summary = self.poi.technology_summary()

    def create_drill_down_dfs(self):
        """ ServiceAggregator 및 POI에 드릴다운 보고서 생성을 지시하는 함수
        """
        if self.opt_engine:
            self.drill_down_dict.update(self.poi.drill_down_dfs(monthly_data=self.monthly_data, time_series_data=self.time_series_data,
                                                                technology_summary=self.technology_summary))
        self.drill_down_dict.update(self.service_agg.drill_down_dfs(monthly_data=self.monthly_data, time_series_data=self.time_series_data,
                                                                    technology_summary=self.technology_summary))
        TellUser.info("Finished post optimization analysis")

    def calculate_cba(self):
        """ 비용 대 이익 분석을 수행하는 함수수
        """
        self.cost_benefit_analysis.calculate(self.poi.der_list, self.service_agg.value_streams, self.time_series_data, self.opt_years)

    def save_as_csv(self, instance_key, sensitivity=False):
        """ 중요한 데이터프레임을 디스크에 CSV 파일로 저장하는 함수
        """
        if sensitivity:
            savepath = self.dir_abs_path / str(instance_key)
        else:
            savepath = self.dir_abs_path
        if not savepath.exists():
            os.makedirs(savepath)

        suffix = f"{self.csv_label}.csv"

        # time series
        self.time_series_data.index.rename('Start Datetime (hb)', inplace=True)
        self.time_series_data.sort_index(axis=1, inplace=True)  # sorts by column name alphabetically
        self.time_series_data.to_csv(path_or_buf=Path(savepath, f'timeseries_results{suffix}'))
        # monthly data
        self.monthly_data.to_csv(path_or_buf=Path(savepath, f'monthly_data{suffix}'))
        # technology summary
        self.technology_summary.to_csv(path_or_buf=Path(savepath, f'technology_summary{suffix}'))

        # save the drill down dfs  NOTE lists are faster to iterate through -- HN
        for file_name, df in self.drill_down_dict.items():
            df.to_csv(path_or_buf=Path(savepath, f"{file_name}{suffix}"))
        # PRINT FINALCIAL/CBA RESULTS
        finacials_dfs = self.cost_benefit_analysis.report_dictionary()
        for file_name, df in finacials_dfs.items():
            df.to_csv(path_or_buf=Path(savepath, f"{file_name}{suffix}"))

        if self.verbose:
            self.objective_values.to_csv(path_or_buf=Path(savepath, f'objective_values{suffix}'))
        TellUser.info(f'Results have been saved to: {savepath}')

    @classmethod
    def sensitivity_summary(cls):
        """ 모든 결과 인스턴스를 반복하면서 감도 분석 결과를 요약한 데이터프레임을 생성하고 CSV 파일로 저장하는 함수
        """
        if cls.sensitivity:
            for key, results_object in cls.instances.items():
                if not key:
                    for npv_col in results_object.cost_benefit_analysis.npv.columns:
                        cls.sensitivity_df.loc[:, npv_col] = 0
                this_npv = results_object.cost_benefit_analysis.npv.reset_index(drop=True, inplace=False)
                if this_npv.empty:
                    # then no optimization ran, so there is no NPV
                    continue
                this_npv.index = pd.RangeIndex(start=key, stop=key + 1, step=1)
                cls.sensitivity_df.update(this_npv)
            cls.sensitivity_df.to_csv(path_or_buf=Path(cls.dir_abs_path, 'sensitivity_summary.csv'))

    @classmethod
    def proforma_df(cls, instance=0):
        """ 특정 인스턴스에 대한 재무 프로포마를 반환하는 함수
        """
        return cls.instances[instance].cost_benefit_analysis.pro_forma
