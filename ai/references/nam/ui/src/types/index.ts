export interface User {
  id?: string;
  username: string;
  link: string;
  server: string;
}

export interface AppsManager {
  [key: string | number]: number;
}
