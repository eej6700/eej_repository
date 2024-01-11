"""runStorageVET.py

StorageVET를 실행하는 초기 실행 지점으로 사용되며
StoregeVET 2.0 또는 SVETpy의 Python 기반 버전을 실행함함
"""

from storagevet.Scenario import Scenario 
                                # 시나리오 처리하는 클래스
                                # POI 및 서비스 집계기
from storagevet.Params import Params
                              # 모델 매개변수를 초기화하는 클래스
                              # 모델 매개변수 CSV 또는 XML 파일에서 사례를 설명하는 Params 객체 초기화함
from storagevet.Result import Result
                              # 결과 처리하는 클래스
                              # 각 시나리오의 실행 결과를 저장하고 분석하는 데 사용함
import time
from storagevet.Visualization import Visualization
                                     # 시각화 처리하는 클래스
                                     # verbose 모드에서 시각화 요약을 출력함
from storagevet.ErrorHandling import *


class StorageVET:
    """ StorageVET API. This will eventually allow StorageVET to be imported and used like any
    other python library.

    """

    def __init__(self, model_parameters_path, verbose=False):
        """ StorageVET을 실행하기 위한 매개변수 및 데이터를 초기화하는 함수
        model_parameters_path는 모델 매개변수 CSV 또는 XML 파일의 경로임
        """
        self.verbose = verbose
        # Initialize the Params Object from Model Parameters
        self.case_dict = Params.initialize(model_parameters_path, verbose)  # unvalidated case instances
        self.results = Result.initialize(Params.results_inputs, Params.case_definitions)
        if verbose:
            self.visualization = Visualization(Params)
            self.visualization.class_summary()

    def solve(self):
        """ StorageVET을 실행하고 결과 반환하는 함수
        각 시나리오에 대해 Scenario를 초기화하고 최적화 문제를 해결한 후 결과를 Result에 추가함
        최종적으로 결과를 반환하고 실행 시간을 출력함
        """
        starts = time.time()
        for key, value in self.case_dict.items():
            run = Scenario(value)
            run.set_up_poi_and_service_aggregator()
            run.initialize_cba()
            run.fill_and_drop_extra_data()
            run.optimize_problem_loop()

            Result.add_instance(key, run)  # cost benefit analysis is in the Result class

        Result.sensitivity_summary()

        ends = time.time()
        TellUser.info("Full runtime: " + str(ends - starts))
        return Result
