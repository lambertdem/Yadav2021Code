using DataFrames, CSV

path = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\"
Matrix{Float64}(CSV.read(string(path,"ObservationsHQ.csv"),DataFrame))