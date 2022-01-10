#install needed packages
echo -e "Installing packages, this might take a while.\n"
pip3 install virtualenv
python3 -m venv env
source env/bin/activate
# source env/bin/activate.fish
pip install -e .
# pip3 install pandas
# pip3 install numpy
# pip3 install matplotlib
# pip3 install uncertainties
echo -e "Running the Skript... \n"
#run the script
python3 src/main.py
