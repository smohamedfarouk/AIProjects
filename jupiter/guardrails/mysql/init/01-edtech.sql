-- Sample edtech database for the text-to-SQL agent (all people and contact details are fictional).
-- Runs automatically the first time the MySQL container starts (docker-entrypoint-initdb.d).

CREATE TABLE courses (
    course_id      INT PRIMARY KEY,
    course_name    VARCHAR(100) NOT NULL,
    duration_weeks INT NOT NULL,
    fee            DECIMAL(10, 2) NOT NULL
);

CREATE TABLE students (
    student_id    INT PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    city          VARCHAR(50) NOT NULL,
    email         VARCHAR(100) NOT NULL,
    phone         VARCHAR(15) NOT NULL,
    date_of_birth DATE NOT NULL,
    joined_on     DATE NOT NULL
);

CREATE TABLE enrollments (
    enrollment_id INT PRIMARY KEY,
    student_id    INT NOT NULL,
    course_id     INT NOT NULL,
    enrolled_on   DATE NOT NULL,
    status        VARCHAR(20) NOT NULL, -- active, completed, dropped
    FOREIGN KEY (student_id) REFERENCES students (student_id),
    FOREIGN KEY (course_id) REFERENCES courses (course_id)
);

CREATE TABLE placements (
    placement_id INT PRIMARY KEY,
    student_id   INT NOT NULL,
    company      VARCHAR(100) NOT NULL,
    role         VARCHAR(100) NOT NULL,
    package_lpa  DECIMAL(5, 2) NOT NULL, -- annual package in lakhs
    placed_on    DATE NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students (student_id)
);

CREATE TABLE payments (
    payment_id INT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id  INT NOT NULL,
    amount     DECIMAL(10, 2) NOT NULL,
    paid_on    DATE NOT NULL,
    method     VARCHAR(20) NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students (student_id),
    FOREIGN KEY (course_id) REFERENCES courses (course_id)
);

INSERT INTO courses VALUES
    (1, 'GenAI',                  16, 85000.00),
    (2, 'Data Science',           24, 75000.00),
    (3, 'Full Stack Development', 20, 60000.00),
    (4, 'Cloud Computing',        12, 55000.00),
    (5, 'Cyber Security',         16, 65000.00);

INSERT INTO students VALUES
    (1,  'Aarav Sharma',    'Bangalore', 'aarav.sharma@example.com',    '9800000001', '2000-03-14', '2025-01-10'),
    (2,  'Diya Iyer',       'Chennai',   'diya.iyer@example.com',       '9800000002', '1999-07-22', '2025-01-12'),
    (3,  'Vihaan Reddy',    'Hyderabad', 'vihaan.reddy@example.com',    '9800000003', '2001-11-05', '2025-02-01'),
    (4,  'Ananya Rao',      'Bangalore', 'ananya.rao@example.com',      '9800000004', '2000-01-30', '2025-02-03'),
    (5,  'Arjun Nair',      'Pune',      'arjun.nair@example.com',      '9800000005', '1998-09-18', '2025-02-15'),
    (6,  'Ishita Gupta',    'Delhi',     'ishita.gupta@example.com',    '9800000006', '2002-04-09', '2025-03-01'),
    (7,  'Kabir Mehta',     'Mumbai',    'kabir.mehta@example.com',     '9800000007', '1999-12-25', '2025-03-04'),
    (8,  'Saanvi Kulkarni', 'Pune',      'saanvi.kulkarni@example.com', '9800000008', '2001-06-11', '2025-03-20'),
    (9,  'Reyansh Das',     'Bangalore', 'reyansh.das@example.com',     '9800000009', '2000-08-02', '2025-04-02'),
    (10, 'Myra Joshi',      'Mumbai',    'myra.joshi@example.com',      '9800000010', '2002-02-17', '2025-04-10'),
    (11, 'Aditya Verma',    'Delhi',     'aditya.verma@example.com',    '9800000011', '1998-05-27', '2025-04-18'),
    (12, 'Kiara Menon',     'Bangalore', 'kiara.menon@example.com',     '9800000012', '2001-10-08', '2025-05-05'),
    (13, 'Rohan Pillai',    'Chennai',   'rohan.pillai@example.com',    '9800000013', '1999-03-03', '2025-05-12'),
    (14, 'Aadhya Singh',    'Hyderabad', 'aadhya.singh@example.com',    '9800000014', '2000-12-12', '2025-06-01'),
    (15, 'Vivaan Patel',    'Bangalore', 'vivaan.patel@example.com',    '9800000015', '2002-07-07', '2025-06-09'),
    (16, 'Anika Bose',      'Delhi',     'anika.bose@example.com',      '9800000016', '2001-01-21', '2025-06-20'),
    (17, 'Krishna Murthy',  'Chennai',   'krishna.murthy@example.com',  '9800000017', '1998-11-29', '2025-07-02'),
    (18, 'Pari Chatterjee', 'Pune',      'pari.chatterjee@example.com', '9800000018', '2000-04-15', '2025-07-15'),
    (19, 'Ayaan Khan',      'Hyderabad', 'ayaan.khan@example.com',      '9800000019', '2001-09-09', '2025-08-01'),
    (20, 'Navya Hegde',     'Bangalore', 'navya.hegde@example.com',     '9800000020', '2002-05-19', '2025-08-10');

INSERT INTO enrollments VALUES
    (1,  1,  1, '2025-01-10', 'completed'),
    (2,  2,  2, '2025-01-12', 'completed'),
    (3,  3,  1, '2025-02-01', 'completed'),
    (4,  4,  2, '2025-02-03', 'completed'),
    (5,  5,  3, '2025-02-15', 'completed'),
    (6,  6,  1, '2025-03-01', 'completed'),
    (7,  7,  4, '2025-03-04', 'completed'),
    (8,  8,  2, '2025-03-20', 'completed'),
    (9,  9,  1, '2025-04-02', 'active'),
    (10, 10, 5, '2025-04-10', 'completed'),
    (11, 11, 2, '2025-04-18', 'dropped'),
    (12, 12, 1, '2025-05-05', 'active'),
    (13, 13, 3, '2025-05-12', 'completed'),
    (14, 14, 2, '2025-06-01', 'active'),
    (15, 15, 1, '2025-06-09', 'active'),
    (16, 16, 4, '2025-06-20', 'active'),
    (17, 17, 5, '2025-07-02', 'active'),
    (18, 18, 2, '2025-07-15', 'active'),
    (19, 19, 3, '2025-08-01', 'active'),
    (20, 20, 1, '2025-08-10', 'active'),
    (21, 1,  4, '2025-06-01', 'completed'),
    (22, 4,  1, '2025-07-01', 'active');

INSERT INTO placements VALUES
    (1,  1,  'Infosys',   'GenAI Engineer',       14.50, '2025-06-15'),
    (2,  2,  'TCS',       'Data Analyst',          8.00, '2025-08-01'),
    (3,  3,  'Microsoft', 'ML Engineer',          32.00, '2025-07-10'),
    (4,  4,  'Flipkart',  'Data Scientist',       22.50, '2025-09-05'),
    (5,  5,  'Wipro',     'Full Stack Developer',  7.20, '2025-07-25'),
    (6,  6,  'Google',    'AI Research Engineer', 41.00, '2025-07-30'),
    (7,  7,  'Accenture', 'Cloud Engineer',        9.50, '2025-06-20'),
    (8,  8,  'Swiggy',    'Data Scientist',       18.00, '2025-10-02'),
    (9,  10, 'Deloitte',  'Security Analyst',     11.00, '2025-08-28'),
    (10, 13, 'Zoho',      'Software Engineer',     8.80, '2025-09-15');

INSERT INTO payments VALUES
    (1,  1,  1, 85000.00, '2025-01-10', 'UPI'),
    (2,  2,  2, 75000.00, '2025-01-12', 'Card'),
    (3,  3,  1, 85000.00, '2025-02-01', 'UPI'),
    (4,  4,  2, 75000.00, '2025-02-03', 'NetBanking'),
    (5,  5,  3, 60000.00, '2025-02-15', 'UPI'),
    (6,  6,  1, 85000.00, '2025-03-01', 'Card'),
    (7,  7,  4, 55000.00, '2025-03-04', 'UPI'),
    (8,  8,  2, 75000.00, '2025-03-20', 'UPI'),
    (9,  9,  1, 42500.00, '2025-04-02', 'Card'),
    (10, 10, 5, 65000.00, '2025-04-10', 'NetBanking'),
    (11, 11, 2, 37500.00, '2025-04-18', 'UPI'),
    (12, 12, 1, 85000.00, '2025-05-05', 'UPI'),
    (13, 13, 3, 60000.00, '2025-05-12', 'Card'),
    (14, 14, 2, 75000.00, '2025-06-01', 'UPI'),
    (15, 15, 1, 42500.00, '2025-06-09', 'UPI'),
    (16, 16, 4, 55000.00, '2025-06-20', 'Card'),
    (17, 17, 5, 65000.00, '2025-07-02', 'UPI'),
    (18, 18, 2, 75000.00, '2025-07-15', 'NetBanking'),
    (19, 19, 3, 60000.00, '2025-08-01', 'UPI'),
    (20, 20, 1, 85000.00, '2025-08-10', 'Card'),
    (21, 1,  4, 55000.00, '2025-06-01', 'UPI'),
    (22, 4,  1, 85000.00, '2025-07-01', 'UPI');
