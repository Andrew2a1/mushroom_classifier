class DataConverter:
    def convert_data(self, data):
        converted = []
        for row in data.values:
            attributes = row[3:]
            new_attribures = [self.attribute_to_number(attr) for attr in attributes]
            converted.append(new_attribures)
        return converted

    def attribute_to_number(self, attribute):
        if type(attribute) == float:
            return -1

        filtered = attribute.replace("[", "").replace("]", "").split(",")
        filtered = [v.strip() for v in filtered]
        values = [self.to_float(v) for v in filtered]
        return sum(values)

    def to_float(self, value):
        try:
            return float(value)
        except ValueError:
            return ord(value)