class AuthSession {
  final String token;
  final String userName;
  final String userEmail;
  final int userId;

  const AuthSession({
    required this.token,
    required this.userName,
    required this.userEmail,
    required this.userId,
  });
}
