from prettytable import PrettyTable
from storagevet.ErrorHandling import *


class Visualization:

    def __init__(self, params_class):
        self.params_class = params_class
             # Params 클래스의 인스턴스를 나타내는 매개변수
             # 시각화 클래스의 초기화에 사용됨

    def class_summary(self):
        """ Model_Parameters_Template을 요약하여 시각화하는 함수
        """
        input_tags = self.params_class.json_tree
        tree = self.params_class.xmlTree
        treeRoot = None
        if tree is not None:
            treeRoot = tree.getroot()
        schema = self.params_class.schema_dct

        TellUser.info("Printing summary table for class Params")
        table = PrettyTable()
        table.field_names = ["Category", "Tag", "ID", "Active?", "Key", "Value", "Type", "Sensitivity?", "Values", "Coupled with?"]
        if input_tags is not None:
            for tag_name, tag_ids in input_tags.items():
                schemaType = self.get_schema_type(schema, tag_name)
                for id_str, tag_id_attrib in tag_ids.items():
                    activeness = tag_id_attrib.get('active')
                    # don't show inactive rows in detail
                    if activeness[0].lower() == "y" or activeness[0] == "1":
                        keys = tag_id_attrib.get('keys')
                        for key_name, key_attrib in keys.items():
                            sensitivity_attrib = key_attrib.get('sensitivity')
                            table.add_row([schemaType, tag_name, id_str, activeness, key_name, key_attrib.get('opt_value'), key_attrib.get('type'),
                                           sensitivity_attrib.get('active'), sensitivity_attrib.get('value'), sensitivity_attrib.get('coupled')])
                    else:
                        table.add_row([schemaType, tag_name, id_str, activeness, '-', '-', '-', '-', '-', '-'])
        else:
            for tag in treeRoot:
                schemaType = self.get_schema_type(schema, tag.tag)
                activeness = tag.get('active')
                id_str = tag.get('id')
                # don't show inactive rows in detail
                if activeness[0].lower() == "y" or activeness[0] == "1":
                    for key in tag:
                        table.add_row([schemaType, tag.tag, id_str, activeness, key.tag, key.find('Value').text, key.find('Type').text,
                                       key.get('analysis'), key.find('Sensitivity_Parameters').text, key.find('Coupled').text])
                else:
                    table.add_row([schemaType, tag.tag, activeness, '-', '-', '-', '-', '-', '-'])

        TellUser.info('User input summary: \n' + str(table))
        return table

    @staticmethod
    def get_schema_type(schema_dict, component_name):
        """ 스키마 XML에서 구성 요소의 유형을 찾는 함수
        클래스 요약을 인쇄하는데 사용되는 함수
        schema_dict: 스키마 딕셔너리
        component_name: 유형을 찾을 속성의 이름
        """
        tag_dicts = schema_dict.get("tags")
        for tag_name, tag_attrib in tag_dicts.items():
            if tag_name == component_name:
                if tag_attrib.get('type') is None:
                    return "other"
                else:
                    return tag_attrib.get('type')
