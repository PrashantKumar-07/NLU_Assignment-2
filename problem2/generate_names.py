import random
from pathlib import Path

MALE_FIRST = [
    "Aarav", "Aakash", "Abhijit", "Abhishek", "Aditya", "Ajay", "Akash",
    "Alok", "Amarjeet", "Amit", "Amol", "Anand", "Anirban", "Anil", "Ankit",
    "Arnab", "Arjun", "Arjunan", "Aryan", "Arvind", "Ashish", "Ashok", "Atul",
    "Balaji", "Baljinder", "Binod", "Chandran", "Chetan", "Daljeet", "Debashis",
    "Debasish", "Deepak", "Dhruv", "Dinakaran", "Dinesh", "Dipankar", "Farhan",
    "Ganesh", "Gaurav", "Girish", "Gopal", "Govind", "Gurpreet", "Harish",
    "Harjinder", "Harsh", "Hemant", "Himanshu", "Indranil", "Indrajit",
    "Jagdish", "Jay", "Jayanta", "Jaswinder", "Kailash", "Kartik", "Karthik",
    "Karthikeyan", "Kiran", "Kunal", "Kulwinder", "Lalit", "Logesh", "Madhav",
    "Mahesh", "Mainak", "Manikandan", "Manish", "Manpreet", "Mayank", "Mohit",
    "Murali", "Narayanan", "Naresh", "Naveen", "Navjot", "Neeraj", "Nikhil",
    "Nitin", "Om", "Palani", "Pankaj", "Paramjit", "Parth", "Partha",
    "Piyush", "Pradeep", "Pranav", "Prasad", "Prashant", "Praveen", "Prithwish",
    "Rahul", "Rajan", "Rajesh", "Rajiv", "Rakesh", "Ram", "Ramakant",
    "Ramandeep", "Ramesh", "Ravi", "Ritesh", "Rohit", "Rupesh", "Sachin",
    "Samir", "Sandeep", "Sanjay", "Saravanan", "Satish", "Saurabh", "Senthil",
    "Shivam", "Shubham", "Siddharth", "Simranjit", "Subhajit", "Subhro",
    "Subramanian", "Sudipto", "Sumit", "Sunil", "Supriyo", "Suresh", "Swarup",
    "Tarun", "Thirumaran", "Trilok", "Tushar", "Uday", "Udhayakumar", "Umesh",
    "Vaibhav", "Vijay", "Vijayakumar", "Vikram", "Vinay", "Vignesh", "Vinoth",
    "Vishal", "Vivek", "Yash", "Yogesh", "Yuvaraj",
    "Abhinav", "Adarsh", "Amitava", "Anshul", "Anupam", "Arpit", "Avijit",
    "Basant", "Bharat", "Bhushan", "Debdut", "Devraj", "Dilip", "Gaurang",
    "Harshit", "Hitesh", "Jatin", "Kamal", "Kapil", "Keshav", "Lokesh",
    "Manas", "Manoj", "Milind", "Mukesh", "Nandan", "Navin", "Nilesh",
    "Paresh", "Priyesh", "Puneet", "Raghav", "Rajat", "Rajendra", "Ramgopal",
    "Ranjit", "Raunak", "Ravindra", "Ritik", "Roopesh", "Rupak", "Sagar",
    "Sameer", "Santosh", "Satyam", "Shailesh", "Shekhar", "Shyam", "Somnath",
    "Sourav", "Sridhar", "Subhash", "Suhas", "Sujit", "Sundar", "Suraj",
    "Swapnil", "Tanmay", "Tejas", "Umang", "Utkarsh", "Venkat", "Vikas",
    "Vipin", "Viren", "Yuvraj", "Zeeshan",
]

FEMALE_FIRST = [
    "Aanya", "Aditi", "Akanksha", "Aishwarya", "Amandeep", "Ambika", "Amrita",
    "Ananya", "Anjali", "Ankita", "Aparna", "Archana", "Arpita", "Asha",
    "Bhavani", "Bhavna", "Chandni", "Chandrika", "Deepa", "Deepika", "Devi",
    "Dipika", "Divya", "Gauri", "Gayathri", "Geeta", "Gurdeep", "Harsimran",
    "Harleen", "Hema", "Indira", "Isha", "Jasleen", "Jasmeet", "Jayanthi",
    "Jyoti", "Kajal", "Kamala", "Kavita", "Kavitha", "Komal", "Lakshmi",
    "Laxmi", "Lopamudra", "Madhuri", "Madhumita", "Malathi", "Mallika",
    "Mamta", "Meena", "Meenakshi", "Meera", "Megha", "Monika", "Moumita",
    "Mythili", "Nalini", "Namita", "Navneet", "Neha", "Nikita", "Nisha",
    "Padma", "Parminder", "Parvathi", "Poonam", "Pooja", "Poulami", "Preeti",
    "Pritha", "Priya", "Rachna", "Radha", "Rajalakshmi", "Ramya", "Rekha",
    "Renu", "Revathi", "Ritika", "Riya", "Rupinder", "Sakshi", "Sanchita",
    "Sandhya", "Sangeeta", "Sanjana", "Sapna", "Saranya", "Savitha", "Seema",
    "Shanthi", "Shikha", "Shreya", "Shruti", "Simran", "Smita", "Sneha",
    "Sona", "Sonal", "Sowmya", "Srabanti", "Subha", "Sudha", "Suganya",
    "Sukhmani", "Sumathi", "Sunita", "Susmita", "Swati", "Tanushree",
    "Tanvi", "Thilagam", "Tiyasha", "Usha", "Vandana", "Varsha", "Vasantha",
    "Vibha", "Vidya", "Vijayalakshmi", "Vimala", "Vineeta", "Yamini",
    "Yukta", "Zara",
    "Akansha", "Alka", "Anita", "Anshu", "Aradhana", "Babita", "Bharati",
    "Charu", "Ekta", "Garima", "Geetanjali", "Gunjan", "Hemlata", "Jaya",
    "Kanchan", "Karuna", "Kiran", "Kriti", "Kusum", "Lalita", "Leela",
    "Mala", "Manisha", "Manju", "Mansi", "Mrinalini", "Nandita", "Neelam",
    "Nidhi", "Pallavi", "Pooja", "Prachi", "Prerna", "Pushpa", "Ragini",
    "Ranjana", "Rashmi", "Reena", "Rohini", "Ruchika", "Rukmini", "Sandhya",
    "Sarita", "Savita", "Shalini", "Shama", "Shobha", "Shweta", "Sita",
    "Smriti", "Sonam", "Sujata", "Sulekha", "Sunaina", "Swapna", "Taruna",
    "Uma", "Vasudha", "Veena", "Vidushi", "Vina", "Vrinda", "Yashoda",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Yadav", "Mishra",
    "Pandey", "Tiwari", "Srivastava", "Dubey", "Shukla", "Dwivedi",
    "Tripathi", "Agarwal", "Bansal", "Jain", "Garg", "Mathur", "Saxena",
    "Chauhan", "Rajput", "Thakur", "Patel", "Shah", "Mehta", "Joshi",
    "Nair", "Pillai", "Menon", "Krishnan", "Rajan", "Subramanian",
    "Venkataraman", "Iyer", "Iyengar", "Naidu", "Reddy", "Rao", "Murthy",
    "Prasad", "Swamy", "Chetty", "Nambiar", "Chatterjee", "Mukherjee",
    "Banerjee", "Ghosh", "Bose", "Sen", "Das", "Roy", "Dutta", "Sarkar",
    "Chakraborty", "Bhattacharya", "Grewal", "Sidhu", "Dhaliwal", "Sandhu",
    "Gill", "Sethi", "Khanna", "Chopra", "Malhotra", "Kapoor", "Arora",
    "Bhatia", "Desai", "Kulkarni", "Patil", "Wagh", "Jadhav", "Shinde",
    "Bhosale", "More", "Pawar", "Gaikwad",
]


def generate_names(n: int = 1000, seed: int = 42) -> list[str]:
    random.seed(seed)
    all_first = list(dict.fromkeys(MALE_FIRST + FEMALE_FIRST))  
    usage = {name: 0 for name in all_first}
    max_reuse = max(1, (n // len(all_first)) + 1)

    names = set()
    attempts = 0

    while len(names) < n and attempts < n * 20:
        attempts += 1

        available = [f for f in all_first if usage[f] < max_reuse]
        if not available:
            max_reuse += 1
            available = all_first

        first = random.choice(available)
        last  = random.choice(LAST_NAMES)

        roll = random.random()
        if roll < 0.20:
            name = first
        elif roll < 0.50:
            name = f"{first} {last}"
        elif roll < 0.75:
            initial = random.choice("ABCDEFGHIJKLMNOPRSTUV")
            name = f"{first} {initial}. {last}"
        else:
            second = random.choice([f for f in all_first if f != first])
            name = f"{first} {second} {last}"

        if name not in names:
            names.add(name)
            usage[first] += 1

    return sorted(names)[:n]


def main():
    out_path = Path(__file__).parent / "TrainingNames.txt"
    names = generate_names(1000)
    with open(out_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
    print(f"Saved {len(names)} names to {out_path}")


if __name__ == "__main__":
    main()